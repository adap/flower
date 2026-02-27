import os
import pickle
import time
from time import time

import flwr as fl
import hydra
import numpy as np
import pandas as pd
import torch
import wandb
from setup.client import generate_client_fn, generate_client_fn_subset
from setup.data_loader import (
    load_shakespeare_subsets_offline,
    load_subsets_offline,
)
from setup.model import (
    CNN,
    AlexNetCIFAR,
    AlexNetFMNIST,
    AlexNetImageNet,
    DenseNet121,
    LeNet5,
    LSTMShakespeare,
    get_parameters,
    vgg19,
)
from setup.server import (
    PrivacyAttacksForDefense,
    PrivacyAttacksForDefenseFedProx,
    get_evaluate_fn,
)


def check_config(cfg, attack_types):
    if "cosine" not in attack_types and "yeom" not in attack_types:
        cfg.canary = False
        cfg.noise = False
    else:
        if cfg.noise:
            if not cfg.canary and not cfg.noise:
                raise ValueError("ALERT: Canary and noise are both false!")
        else:
            cfg.canary = True
            cfg.dynamic_canary = True
            cfg.single_training = False

    if cfg.dataset == "cifar100":
        cfg.num_classes = 100
        cfg.image_label = "fine_label"
        cfg.image_name = "img"
        cfg.input_size = 32

    elif cfg.dataset == "cifar10":
        cfg.num_classes = 10
        cfg.image_label = "label"
        cfg.image_name = "img"
        cfg.input_size = 32

    elif cfg.dataset == "fmnist":
        cfg.num_classes = 10
        cfg.image_label = "label"
        cfg.image_name = "img"
        cfg.input_size = 28

    elif cfg.dataset == "shakespeare":
        # Load vocabulary to get exact vocab size
        try:
            vocab_path = f"{cfg.path_to_local_dataset}vocab.pkl"
            with open(vocab_path, "rb") as f:
                vocab_info = pickle.load(f)
            cfg.vocab_size = vocab_info["vocab_size"]
            print(f"Loaded vocabulary: {cfg.vocab_size} characters")
        except:
            # Fallback to config value
            cfg.vocab_size = cfg.get("vocab_size", 65)
            print(f"Using vocab_size from config: {cfg.vocab_size}")

        cfg.num_classes = cfg.vocab_size  # For character prediction
        cfg.text_label = "target"
        cfg.text_input = "input"
        cfg.seq_length = cfg.get("seq_length", 80)
        print(
            f"Shakespeare config: vocab_size={cfg.vocab_size}, seq_length={cfg.seq_length}"
        )

    return cfg


def get_model(cfg, device):
    print(
        f"Architecture: {cfg.architecture}, Dataset: {cfg.dataset}, Num classes: {cfg.num_classes}"
    )
    if cfg.architecture == "AlexNet" and cfg.dataset in ["cifar10", "cifar100"]:
        return AlexNetCIFAR(cfg.num_classes, device)

    elif cfg.architecture == "AlexNet" and cfg.dataset == "fmnist":
        return AlexNetFMNIST(cfg.num_classes, device)

    elif cfg.architecture == "AlexNet" and cfg.dataset == "imagenet":
        return AlexNetImageNet(cfg.num_classes, device)

    elif cfg.architecture == "LeNet5":
        return LeNet5(cfg.num_classes, device)

    elif cfg.architecture == "CNN":
        return CNN(cfg, device)

    elif cfg.architecture == "VGG":
        return vgg19()

    elif cfg.architecture == "DenseNet":
        return DenseNet121(num_classes=cfg.num_classes, grayscale=False)

    elif cfg.architecture == "LSTM" and cfg.dataset == "shakespeare":
        return LSTMShakespeare(
            vocab_size=cfg.vocab_size,
            embedding_dim=cfg.get("embedding_dim", 8),
            hidden_dim=cfg.get("hidden_dim", 100),
            num_layers=cfg.get("num_layers", 2),
            num_classes=cfg.num_classes,
            device=device,
        )
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")


def setup_simulation(cfg, subset_loader_fn, strategy_class=PrivacyAttacksForDefense):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time.sleep(0.1)

    cfg = check_config(cfg, cfg.attack_types)
    model = get_model(cfg, DEVICE).to(DEVICE)

    params = get_parameters(model)
    print("Params: {}".format(np.concatenate([arr.flatten() for arr in params]).shape))
    freeriders = [
        num_client < cfg.num_freeriders for num_client in range(cfg.num_clients)
    ]
    print("Free-riders: {}".format(freeriders))

    wandb.init(
        mode=cfg.wandb_mode,
        project=cfg.wandb_project,
        config={
            "learning_rate": cfg.lr,
            "architecture": cfg.architecture,
            "dataset": cfg.dataset,
            "rounds": cfg.num_rounds,
            "attack_types": cfg.attack_types,
            "num_freeriders": cfg.num_freeriders,
            "freerider_type": cfg.freerider_type,
            "num_clients": cfg.num_clients,
            "batch_size": cfg.batch_size,
            "subset_samples": cfg.subset_samples,
            "multiplicator": cfg.multiplicator,
            "pia_type": cfg.pia_type,
            "iid": cfg.iid,
            "dirichlet_alpha": cfg.dirichlet_alpha,
            "freerider_canary": cfg.freerider_canary,
            "vocab_size": cfg.get("vocab_size", None),
            "seq_length": cfg.get("seq_length", None),
            "proximal_mu": cfg.get("proximal_mu", 0.0),  # Add this line
        },
    )

    if cfg.canary:
        trainloaders, valloaders, subsetloaders = subset_loader_fn(cfg)
        client_fn = generate_client_fn_subset(
            model, trainloaders, valloaders, freeriders, subsetloaders, cfg, DEVICE
        )
    else:
        trainloaders, valloaders, _ = subset_loader_fn(cfg)
        client_fn = generate_client_fn(
            model, trainloaders, valloaders, freeriders, cfg, DEVICE
        )

    # FedProx: Add proximal_mu parameter if using FedProx strategy
    strategy_kwargs = {
        "fraction_fit": 0.00001,
        "fraction_evaluate": 0.0,
        "min_fit_clients": cfg.num_clients,
        "min_available_clients": cfg.num_clients,
        "initial_parameters": fl.common.ndarrays_to_parameters(params),
        "evaluate_fn": get_evaluate_fn(model, valloaders, DEVICE, cfg),
        "training_datasets": trainloaders,
        "validation_datasets": valloaders,
        "net": model,
        "freeriders": freeriders,
        "device": DEVICE,
        "subsets": subsetloaders if cfg.canary else None,
        "cfg": cfg,
    }

    # Add proximal_mu if using FedProx strategy
    if strategy_class == PrivacyAttacksForDefenseFedProx:
        strategy_kwargs["proximal_mu"] = cfg.get("proximal_mu", 0.1)

    strategy = strategy_class(**strategy_kwargs)

    return client_fn, strategy


def format_line(wandb_dir, cfg, extra_params=None):
    """Formats the line to be written to the results file."""
    base_params = {
        "wandb_dir": wandb_dir,
        "num_average_rounds": cfg["average_detection"],
        "num_clients": cfg["num_clients"],
        "num_fr": cfg["num_freeriders"],
        "dataset": cfg["dataset"],
        "fr_type": cfg["freerider_type"],
        "multiplicator": cfg["multiplicator"],
        "power": cfg["power"],
        "iid": cfg["iid"],
        "dirichlet_alpha": cfg["dirichlet_alpha"],
        "attack_types": cfg["attack_types"],
        "use_fedprox": cfg["use_fedprox"],
        "proximal_mu": cfg["proximal_mu"],
        "freerider_canary": cfg["freerider_canary"],
        "architecture": cfg["architecture"],
        "local_epochs": cfg["local_epochs"],
        "dp": cfg["dp"],
        "epsilon": cfg["epsilon"],
        "sigma": cfg["sigma"],
        "zscore_threshold": cfg["zscore_threshold"],
    }
    if extra_params:
        base_params.update(extra_params)
    return ", ".join(f"{k}: {v}" for k, v in base_params.items()) + " \n"


def get_results_path(base_folder, attack_type, cfg, suffix="experiment"):
    """Generates the results file path based on configuration."""
    subfolder = os.path.join(base_folder, attack_type, cfg["freerider_type"])
    os.makedirs(subfolder, exist_ok=True)
    filename = (
        f"{cfg['canary_epochs']}canaryrounds.txt"
        if attack_type in ["cosine", "yeom"]
        else f"{suffix}.txt"
    )
    return os.path.join(subfolder, filename)


def save_experiment_results(cfg, history, wandb_dir):
    """Saves the experimental results based on the configuration."""
    attack_type = (
        "".join(cfg["attack_types"])
        if len(cfg["attack_types"]) > 1
        else cfg["attack_types"][0]
    )
    results_path = get_results_path(cfg["experiments_track_folder"], attack_type, cfg)

    # Prepare parameters specific to the attack type
    extra_params = {}
    if attack_type == "cosine":
        extra_params["canary_epochs"] = cfg["canary_epochs"]
        extra_params["subset_samples"] = cfg["subset_samples"]

    elif attack_type == "yeom":
        extra_params.update(
            {
                "canary_epochs": cfg["canary_epochs"],
                "server_epochs": cfg["server_epochs"],
                "yeom_type": cfg["yeom_type"],
                "subset_samples": cfg["subset_samples"],
            }
        )
    elif attack_type in ["dist_score", "inconsistency"]:
        extra_params["pia_type"] = cfg["pia_type"]

    # Format and write the line
    line_to_append = format_line(wandb_dir, cfg, extra_params)
    with open(results_path, "a") as file:
        file.write(line_to_append)

    # Optional CSV saving
    if attack_type in ["pia_csv", "dist_score", "inconsistency"] and cfg.get(
        "csv", False
    ):
        save_csv(history, cfg)


def run_simulation(cfg, subset_loader_fn=None, strategy_class=PrivacyAttacksForDefense):
    """Main simulation runner."""
    client_fn, strategy = setup_simulation(cfg, subset_loader_fn, strategy_class)
    print("Set-up simulation done")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg["num_clients"],
        config=fl.server.ServerConfig(num_rounds=cfg["num_rounds"]),
        strategy=strategy,
        client_resources={"num_gpus": 1} if torch.cuda.is_available() else None,
    )

    if cfg.get("store", False):
        save_results(history)

    wandb_dir = wandb.run.dir
    if "experiments_track_folder" in cfg:
        save_experiment_results(cfg, history, wandb_dir)

    # Save general experiment metadata
    metadata_path = os.path.join(wandb_dir, "experiments.txt")
    line_to_append = format_line(wandb_dir, cfg)
    with open(metadata_path, "a") as file:
        file.write(line_to_append)


def save_results(history):
    results_path = os.path.join(wandb.run.dir, "results.pkl")
    results = {"history": history}

    with open(str(results_path), "wb") as h:
        pickle.dump(results, h, protocol=pickle.HIGHEST_PROTOCOL)


def save_csv(history, cfg):
    df = None
    for round, data in history.metrics_distributed_fit["pia"]:
        for attack_type, labels in data.items():
            for id, client in enumerate(labels):
                result = {
                    f"L{i}": [label_count] for i, label_count in enumerate(client)
                }
                result["Pia"] = attack_type
                result["FreeRider"] = cfg.num_freeriders > id
                result["Round"] = round
                result["Noise"] = cfg.multiplicator
                result["CID"] = id
                result["IID"] = cfg.iid
                result["Run"] = wandb.run.dir
                result["NumClients"] = cfg.num_clients
                result["FreeRiderType"] = cfg.freerider_type
                result["NumFreeRider"] = cfg.num_freeriders
                result["Architecture"] = cfg.architecture
                result["Dataset"] = cfg.dataset
                result["DirichletAlpha"] = cfg.dirichlet_alpha

                result = pd.DataFrame.from_dict(result)
                if df is None:
                    df = result
                else:
                    df = pd.concat([df, result], ignore_index=True)


@hydra.main(
    config_path="./configurations", config_name="baseline.yaml", version_base=None
)
def simulation(cfg):
    """Main simulation function"""
    print("Start simulation")
    cfg = check_config(cfg, cfg.attack_types)

    if cfg.dataset == "shakespeare":
        subset_loader = load_shakespeare_subsets_offline
    else:
        subset_loader = load_subsets_offline

    # Choose strategy based on config
    strategy_class = (
        PrivacyAttacksForDefenseFedProx
        if cfg.get("use_fedprox", False)
        else PrivacyAttacksForDefense
    )

    run_simulation(cfg, subset_loader_fn=subset_loader, strategy_class=strategy_class)


if __name__ == "__main__":
    simulation()
