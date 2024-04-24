"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""

import functools
from typing import Dict, Union

import flwr as fl
import hydra
import torch
import wandb
from flwr.common import Scalar
from flwr.server.app import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset

from hfedxgboost.client import FlClient
from hfedxgboost.dataset import divide_dataset_between_clients, load_single_dataset
from hfedxgboost.server import FlServer, serverside_eval
from hfedxgboost.utils import (
    CentralizedResultsWriter,
    EarlyStop,
    ResultsWriter,
    create_res_csv,
    local_clients_performance,
    run_centralized,
)


@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the baseline.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # 1. Print parsed config
    print(OmegaConf.to_yaml(cfg))
    writer: Union[ResultsWriter, CentralizedResultsWriter]
    if cfg.centralized:
        if cfg.dataset.dataset_name == "all":
            run_centralized(cfg, dataset_name=cfg.dataset.dataset_name)
        else:
            writer = CentralizedResultsWriter(cfg)
            create_res_csv("results_centralized.csv", writer.fields)
            writer.write_res(
                "results_centralized.csv",
                run_centralized(cfg, dataset_name=cfg.dataset.dataset_name)[0],
                run_centralized(cfg, dataset_name=cfg.dataset.dataset_name)[1],
            )
    else:
        if cfg.use_wandb:
            wandb.init(**cfg.wandb.setup, group=f"{cfg.dataset.dataset_name}")

        print("Dataset Name", cfg.dataset.dataset_name)
        early_stopper = EarlyStop(cfg)
        x_train, y_train, x_test, y_test = load_single_dataset(
            cfg.dataset.task.task_type,
            cfg.dataset.dataset_name,
            train_ratio=cfg.dataset.train_ratio,
        )

        trainloaders, valloaders, testloader = divide_dataset_between_clients(
            TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
            TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test)),
            batch_size=cfg.batch_size,
            pool_size=cfg.clients.client_num,
            val_ratio=cfg.val_ratio,
        )
        print(
            f"Data partitioned across {cfg.clients.client_num} clients"
            f" and {cfg.val_ratio} of local dataset reserved for validation."
        )
        if cfg.show_each_client_performance_on_its_local_data:
            local_clients_performance(
                cfg, trainloaders, x_test, y_test, cfg.dataset.task.task_type
            )

        # Configure the strategy
        def fit_config(server_round: int) -> Dict[str, Scalar]:
            print(f"Configuring round {server_round}")
            return {
                "num_iterations": cfg.run_experiment.fit_config.num_iterations,
                "batch_size": cfg.run_experiment.batch_size,
            }

        # FedXgbNnAvg
        strategy = instantiate(
            cfg.strategy,
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=(
                lambda r: {"batch_size": cfg.run_experiment.batch_size}
            ),
            evaluate_fn=functools.partial(
                serverside_eval,
                cfg=cfg,
                testloader=testloader,
            ),
        )

        print(
            f"FL experiment configured for {cfg.run_experiment.num_rounds} rounds with",
            f"{cfg.clients.client_num} client in the pool.",
        )

        def client_fn(cid: str) -> fl.client.Client:
            """Create a federated learning client."""
            return FlClient(cfg, trainloaders[int(cid)], valloaders[int(cid)], cid)

        # Start the simulation
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            server=FlServer(
                cfg=cfg,
                client_manager=SimpleClientManager(),
                early_stopper=early_stopper,
                strategy=strategy,
            ),
            num_clients=cfg.clients.client_num,
            client_resources=cfg.client_resources,
            config=ServerConfig(num_rounds=cfg.run_experiment.num_rounds),
            strategy=strategy,
        )

        print(history)
        writer = ResultsWriter(cfg)
        print(
            "Best Result",
            writer.extract_best_res(history)[0],
            "best_res_round",
            writer.extract_best_res(history)[1],
        )
        create_res_csv("results.csv", writer.fields)
        writer.write_res("results.csv")


if __name__ == "__main__":
    main()
