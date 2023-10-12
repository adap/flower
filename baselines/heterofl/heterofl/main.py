"""Runs federated learning for given configuration in base.yaml."""
import pickle
from pathlib import Path

import flwr as fl
import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from heterofl import client, models, server
from heterofl.client_manager_heterofl import ClientManagerHeteroFL
from heterofl.dataset import load_datasets
from heterofl.strategy import HeteroFL
from heterofl.utils import ModelRateManager, get_global_model_rate, preprocess_input


@hydra.main(config_path="conf", config_name="base.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))
    torch.manual_seed(cfg.seed)

    data_loaders = {}

    (
        data_loaders["entire_trainloader"],
        data_loaders["trainloaders"],
        data_loaders["label_split"],
        data_loaders["valloaders"],
        data_loaders["testloader"],
    ) = load_datasets(config=cfg.dataset, num_clients=cfg.num_clients, seed=cfg.seed)

    model_config = preprocess_input(cfg.model, cfg.dataset)

    # send this array(client_model_rate_mapping) as
    # an argument to client_manager and client
    model_split_rate = {"a": 1, "b": 0.5, "c": 0.25, "d": 0.125, "e": 0.0625}
    # model_split_mode = cfg.control.model_split_mode
    model_mode = cfg.control.model_mode

    client_to_model_rate_mapping = [float(0) for _ in range(cfg.num_clients)]
    model_rate_manager = ModelRateManager(
        cfg.control.model_split_mode, model_split_rate, model_mode
    )
    # client_manager = ClientManagerHeteroFL(
    #     model_rate_manager,
    #     client_to_model_rate_mapping,
    #     client_label_split=data_loaders["label_split"],
    # )

    model_config["global_model_rate"] = model_split_rate[
        get_global_model_rate(model_mode)
    ]
    test_model = models.create_model(
        model_config,
        model_rate=model_split_rate[get_global_model_rate(model_mode)],
        track=False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    )

    # # for i in range(cfg.num_clients):
    #     # client_to_model_rate_mapping[i]

    # prepare function that will be used to spawn each client
    client_train_settings = {
        "epochs": cfg.num_epochs,
        "optimizer": cfg.strategy.optimizer,
        "lr": cfg.strategy.lr,
        "momentum": cfg.strategy.momentum,
        "weight_decay": cfg.strategy.weight_decay,
        "scheduler": cfg.strategy.scheduler,
        "milestones": cfg.strategy.milestones,
    }

    optim_scheduler_settings = {
        "optimizer": cfg.strategy.optimizer,
        "lr": cfg.strategy.lr,
        "momentum": cfg.strategy.momentum,
        "weight_decay": cfg.strategy.weight_decay,
        "scheduler": cfg.strategy.scheduler,
        "milestones": cfg.strategy.milestones,
    }

    client_fn = client.gen_client_fn(
        model_config=model_config,
        client_to_model_rate_mapping=client_to_model_rate_mapping,
        client_train_settings=client_train_settings,
        data_loaders=data_loaders,
    )

    strategy = HeteroFL(
        model_name=cfg.model.model_name,
        net=models.create_model(
            model_config,
            model_rate=model_split_rate[get_global_model_rate(model_mode)],
            device=cfg.server_device,
        ),
        optim_scheduler_settings=optim_scheduler_settings,
        global_model_rate=model_split_rate[get_global_model_rate(model_mode)],
        evaluate_fn=server.gen_evaluate_fn(
            data_loaders,
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
            test_model,
        ),
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=cfg.num_clients,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
        client_manager=ClientManagerHeteroFL(
            model_rate_manager,
            client_to_model_rate_mapping,
            client_label_split=data_loaders["label_split"],
        ),
        strategy=strategy,
    )

    # save the results
    save_path = HydraConfig.get().runtime.output_dir
    # results_path = Path(save_path) / "results.pkl"
    # model_path = Path(save_path) / "model.pth"
    # results = {"history": history}

    # save the results as a python pickle
    with open(str(Path(save_path) / "results.pkl"), "wb") as file_handle:
        pickle.dump({"history": history}, file_handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save the model
    torch.save(test_model.state_dict(), str(Path(save_path) / "model.pth"))

    # plot grpahs using history and save the results.

    # fl.server.strategy.fedavg
    # fl.simulation.start_simulation
    # fl.server.start_server


if __name__ == "__main__":
    main()
