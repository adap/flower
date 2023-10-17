"""Create and connect the building blocks for your experiments; start the simulation.

It includes processioning the dataset, instantiate strategy, specify how the global
model is going to be evaluated, etc. At the end, this script saves the results.
"""
import functools
from typing import Dict, Union

import flwr as fl
import hydra
import omegaconf
import torch
from flwr.common import Scalar
from flwr.server.app import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset

import wandb
from hfedxgboost.client import FL_Client
from hfedxgboost.dataset import divide_dataset_between_clients, load_single_dataset
from hfedxgboost.server import FL_Server, serverside_eval
from hfedxgboost.utils import (
    Early_Stop,
    clients_performance_on_local_data,
    results_writer,
    results_writer_centralized,
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
    writer: Union[results_writer, results_writer_centralized]
    if cfg.centralized:
        if cfg.dataset.dataset_name == "all":
            run_centralized(cfg, dataset_name=cfg.dataset.dataset_name)
        else:
            result_train, result_test = run_centralized(
                cfg, dataset_name=cfg.dataset.dataset_name
            )
            writer = results_writer_centralized(cfg)
            writer.create_res_csv("results_centralized.csv")
            writer.write_res("results_centralized.csv", result_train, result_test)
    else:

        def run_fed(cfg: DictConfig) -> None:
            print("Dataset Name", cfg.dataset.dataset_name)
            dataset_name = cfg.dataset.dataset_name
            task_type = cfg.dataset.task.task_type
            early_stopper = Early_Stop(cfg)
            X_train, y_train, X_test, y_test = load_single_dataset(
                task_type, dataset_name, train_ratio=cfg.dataset.train_ratio
            )
            print("Feature dimension of the dataset:", X_train.shape[1])
            print("Size of the trainset:", X_train.shape[0])
            print("Size of the testset:", X_test.shape[0])
            if task_type == "BINARY":
                print(
                    "First class ratio in train data",
                    y_train[y_train == 0.0].size / X_train.shape[0],
                )
                print(
                    "Second class ratio in train data",
                    y_train[y_train != 0.0].size / X_train.shape[0],
                )
                print(
                    "First class ratio in test data",
                    y_test[y_test == 0.0].size / X_test.shape[0],
                )
                print(
                    "Second class ratio in test data",
                    y_test[y_test != 0.0].size / X_test.shape[0],
                )
            trainset = TensorDataset(
                torch.from_numpy(X_train), torch.from_numpy(y_train)
            )
            testset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

            trainloaders, valloaders, testloader = divide_dataset_between_clients(
                trainset,
                testset,
                batch_size=cfg.batch_size,
                pool_size=cfg.client_num,
                val_ratio=cfg.val_ratio,
            )
            print(
                f"Data partitioned across {cfg.client_num} clients"
                f" and {cfg.val_ratio} of local dataset reserved for validation."
            )
            if cfg.show_each_client_performance_on_its_local_data:
                clients_performance_on_local_data(
                    cfg, trainloaders, X_test, y_test, task_type
                )

            num_rounds = cfg.run_experiment.num_rounds
            client_pool_size = cfg.client_num
            batch_size = cfg.run_experiment.batch_size

            # Configure the strategy
            def fit_config(server_round: int) -> Dict[str, Scalar]:
                print(f"Configuring round {server_round}")
                return {
                    "num_iterations": cfg.run_experiment.fit_config.num_iterations,
                    "batch_size": batch_size,
                }

            # FedXgbNnAvg
            strategy = instantiate(
                cfg.strategy,
                on_fit_config_fn=fit_config,
                on_evaluate_config_fn=(lambda r: {"batch_size": batch_size}),
                evaluate_fn=functools.partial(
                    serverside_eval,
                    cfg=cfg,
                    testloader=testloader,
                    batch_size=batch_size,
                ),
            )

            print(
                f"FL experiment configured for {num_rounds} rounds with",
                f"{client_pool_size} client in the pool.",
            )

            def client_fn(cid: str) -> fl.client.Client:
                """Create a federated learning client."""
                return FL_Client(
                    cfg,
                    trainloaders[int(cid)],
                    valloaders[int(cid)],
                    client_pool_size,
                    cid,
                    log_progress=False,
                )

            # Start the simulation
            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                server=FL_Server(
                    cfg=cfg,
                    client_manager=SimpleClientManager(),
                    early_stopper=early_stopper,
                    strategy=strategy,
                ),
                num_clients=client_pool_size,
                client_resources=cfg.client_resources,
                config=ServerConfig(num_rounds=num_rounds),
                strategy=strategy,
            )

            print(history)
            writer = results_writer(cfg)
            best_res, best_res_round = writer.extract_best_res(history)
            print("Best Result", best_res, "best_res_round", best_res_round)
            writer.create_res_csv("results.csv")
            writer.write_res("results.csv")

        if cfg.use_wandb:
            wandb_cfg = omegaconf.OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
            group_key = f"{cfg.dataset.dataset_name}"
            with wandb.init(**cfg.wandb.setup, group=group_key, config=wandb_cfg):
                run_fed(cfg)
        else:
            run_fed(cfg)


if __name__ == "__main__":
    main()
