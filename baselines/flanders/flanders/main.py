"""FLANDERS main scrip."""

import importlib
import os
import random
import shutil

import flwr as fl
import hydra
import numpy as np
import pandas as pd
import torch
from flwr.server.client_manager import SimpleClientManager
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .attacks import fang_attack, gaussian_attack, lie_attack, minmax_attack, no_attack
from .client import FMnistClient, MnistClient
from .dataset import do_fl_partitioning, get_fmnist, get_mnist
from .server import EnhancedServer
from .utils import fmnist_evaluate, l2_norm, mnist_evaluate


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 0. Set random seed
    seed = cfg.seed
    np.random.seed(seed)
    np.random.set_state(
        np.random.RandomState(seed).get_state()  # pylint: disable=no-member
    )
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))

    # Skip if:
    # - strategy = bulyan and num_malicious > 20
    # - attack_fn != gaussian and num_malicious = 0
    if cfg.strategy.name == "bulyan" and cfg.server.num_malicious > 20:
        print(
            "Skipping experiment because strategy is bulyan and num_malicious is > 20"
        )
        return
    # skip if attack_fn is not gaussian and num_malicious is 0, but continue if
    # attack_fn is na
    if (
        cfg.server.attack_fn != "gaussian"
        and cfg.server.num_malicious == 0
        and cfg.server.attack_fn != "na"
    ):
        print(
            "Skipping experiment because attack_fn is not gaussian and "
            "num_malicious is 0"
        )
        return

    attacks = {
        "na": no_attack,
        "gaussian": gaussian_attack,
        "lie": lie_attack,
        "fang": fang_attack,  # OPT
        "minmax": minmax_attack,  # AGR-MM
    }

    clients = {
        "mnist": (MnistClient, mnist_evaluate),
        "fmnist": (FMnistClient, fmnist_evaluate),
    }

    # Delete old client_params
    if os.path.exists(cfg.server.history_dir):
        shutil.rmtree(cfg.server.history_dir)

    dataset_name = cfg.dataset
    attack_fn = cfg.server.attack_fn
    num_malicious = cfg.server.num_malicious

    # 2. Prepare your dataset
    if dataset_name in ["mnist", "fmnist"]:
        if dataset_name == "mnist":
            train_path, _ = get_mnist()
        elif dataset_name == "fmnist":
            train_path, _ = get_fmnist()
        fed_dir = do_fl_partitioning(
            train_path,
            pool_size=cfg.server.pool_size,
            alpha=cfg.server.noniidness,
            num_classes=10,
            val_ratio=0.2,
            seed=seed,
        )
    else:
        raise ValueError("Dataset not supported")

    # 3. Define your clients
    # pylint: disable=no-else-return
    def client_fn(cid: str, dataset_name: str = dataset_name):
        client = clients[dataset_name][0]
        if dataset_name in ["mnist", "fmnist"]:
            return client(cid, fed_dir)
        else:
            raise ValueError("Dataset not supported")

    # 4. Define your strategy
    strategy = None
    if cfg.strategy.name == "flanders":
        function_path = cfg.aggregate_fn.aggregate_fn.function
        module_name, function_name = function_path.rsplit(".", 1)
        module = importlib.import_module(module_name, package=__package__)
        aggregation_fn = getattr(module, function_name)

        strategy = instantiate(
            cfg.strategy.strategy,
            evaluate_fn=clients[dataset_name][1],
            on_fit_config_fn=fit_config,
            fraction_fit=1,
            fraction_evaluate=0,
            min_fit_clients=cfg.server.pool_size,
            min_evaluate_clients=0,
            num_clients_to_keep=cfg.server.pool_size - num_malicious,
            aggregate_fn=aggregation_fn,
            aggregate_parameters=cfg.aggregate_fn.aggregate_fn.parameters,
            min_available_clients=cfg.server.pool_size,
            window=cfg.server.warmup_rounds,
            distance_function=l2_norm,
            maxiter=cfg.strategy.strategy.maxiter,
            alpha=cfg.strategy.strategy.alpha,
            beta=int(cfg.strategy.strategy.beta),
        )
    elif cfg.strategy.name == "krum":
        strategy = instantiate(
            cfg.strategy.strategy,
            evaluate_fn=clients[dataset_name][1],
            on_fit_config_fn=fit_config,
            fraction_fit=1,
            fraction_evaluate=0,
            min_fit_clients=cfg.server.pool_size,
            min_evaluate_clients=0,
            num_clients_to_keep=cfg.strategy.strategy.num_clients_to_keep,
            min_available_clients=cfg.server.pool_size,
            num_malicious_clients=num_malicious,
        )
    elif cfg.strategy.name == "dnc":
        strategy = instantiate(
            cfg.strategy.strategy,
            evaluate_fn=clients[dataset_name][1],
            on_fit_config_fn=fit_config,
            fraction_fit=1,
            fraction_evaluate=0,
            min_fit_clients=cfg.server.pool_size,
            min_evaluate_clients=0,
            min_available_clients=cfg.server.pool_size,
            c=cfg.strategy.strategy.c,
            niters=cfg.strategy.strategy.niters,
            num_malicious_clients=num_malicious,
        )
    elif cfg.strategy.name == "fedavg":
        strategy = instantiate(
            cfg.strategy.strategy,
            evaluate_fn=clients[dataset_name][1],
            on_fit_config_fn=fit_config,
            fraction_fit=1,
            fraction_evaluate=0,
            min_fit_clients=cfg.server.pool_size,
            min_evaluate_clients=0,
            min_available_clients=cfg.server.pool_size,
        )
    elif cfg.strategy.name == "bulyan":
        # Get aggregation rule function
        strategy = instantiate(
            cfg.strategy.strategy,
            evaluate_fn=clients[dataset_name][1],
            on_fit_config_fn=fit_config,
            fraction_fit=1,
            fraction_evaluate=0,
            min_fit_clients=cfg.server.pool_size,
            min_evaluate_clients=0,
            min_available_clients=cfg.server.pool_size,
            num_malicious_clients=num_malicious,
            to_keep=cfg.strategy.strategy.to_keep,
        )
    elif cfg.strategy.name == "trimmedmean":
        strategy = instantiate(
            cfg.strategy.strategy,
            evaluate_fn=clients[dataset_name][1],
            on_fit_config_fn=fit_config,
            fraction_fit=1,
            fraction_evaluate=0,
            min_fit_clients=cfg.server.pool_size,
            min_evaluate_clients=0,
            min_available_clients=cfg.server.pool_size,
            beta=cfg.strategy.strategy.beta,
        )
    elif cfg.strategy.name == "fedmedian":
        strategy = instantiate(
            cfg.strategy.strategy,
            evaluate_fn=clients[dataset_name][1],
            on_fit_config_fn=fit_config,
            fraction_fit=1,
            fraction_evaluate=0,
            min_fit_clients=cfg.server.pool_size,
            min_evaluate_clients=0,
            min_available_clients=cfg.server.pool_size,
        )
    else:
        raise ValueError("Strategy not supported")

    # 5. Start Simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.server.pool_size,
        client_resources=cfg.client_resources,
        server=EnhancedServer(
            warmup_rounds=cfg.server.warmup_rounds,
            num_malicious=num_malicious,
            attack_fn=attacks[attack_fn],  # type: ignore
            magnitude=cfg.server.magnitude,
            client_manager=SimpleClientManager(),
            strategy=strategy,
            sampling=cfg.server.sampling,
            history_dir=cfg.server.history_dir,
            dataset_name=dataset_name,
            threshold=cfg.server.threshold,
            omniscent=cfg.server.omniscent,
        ),
        config=fl.server.ServerConfig(num_rounds=cfg.server.num_rounds),
        strategy=strategy,
    )

    save_path = HydraConfig.get().runtime.output_dir

    rounds, test_loss = zip(*history.losses_centralized)
    _, test_accuracy = zip(*history.metrics_centralized["accuracy"])
    _, test_auc = zip(*history.metrics_centralized["auc"])
    _, truep = zip(*history.metrics_centralized["TP"])
    _, truen = zip(*history.metrics_centralized["TN"])
    _, falsep = zip(*history.metrics_centralized["FP"])
    _, falsen = zip(*history.metrics_centralized["FN"])

    path_to_save = [os.path.join(save_path, "results.csv"), "outputs/all_results.csv"]

    for file_name in path_to_save:
        data = pd.DataFrame(
            {
                "round": rounds,
                "loss": test_loss,
                "accuracy": test_accuracy,
                "auc": test_auc,
                "TP": truep,
                "TN": truen,
                "FP": falsep,
                "FN": falsen,
                "attack_fn": [attack_fn for _ in range(len(rounds))],
                "dataset_name": [dataset_name for _ in range(len(rounds))],
                "num_malicious": [num_malicious for _ in range(len(rounds))],
                "strategy": [cfg.strategy.name for _ in range(len(rounds))],
                "aggregate_fn": [
                    cfg.aggregate_fn.aggregate_fn.function for _ in range(len(rounds))
                ],
            }
        )
        if os.path.exists(file_name):
            data.to_csv(file_name, mode="a", header=False, index=False)
        else:
            data.to_csv(file_name, index=False, header=True)


# pylint: disable=unused-argument
def fit_config(server_round):
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": 1,  # number of local epochs
        "batch_size": 32,
    }
    return config


if __name__ == "__main__":
    main()
