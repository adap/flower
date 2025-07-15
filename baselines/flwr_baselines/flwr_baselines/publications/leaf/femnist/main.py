"""Main module for running FEMNIST experiments."""

import pathlib
from functools import partial
from typing import Type, Union

import flwr as fl
import hydra
import pandas as pd
import torch
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig

from flwr_baselines.publications.leaf.femnist.client import create_client
from flwr_baselines.publications.leaf.femnist.dataset.dataset import (
    create_federated_dataloaders,
)
from flwr_baselines.publications.leaf.femnist.strategy import FedAvgSameClients
from flwr_baselines.publications.leaf.femnist.utils import setup_seed, weighted_average


# pylint: disable=too-many-locals
@hydra.main(config_path="conf", version_base=None)
def main(cfg: DictConfig):
    """Main function for running FEMNIST experiments."""
    # Ensure reproducibility
    setup_seed(cfg.random_seed)
    # Specify PyTorch device
    # pylint: disable=no-member
    device = torch.device(cfg.device)
    # Create datasets for federated learning
    trainloaders, valloaders, testloaders = create_federated_dataloaders(
        cfg.dataset.distribution_type,
        cfg.dataset.dataset_fraction,
        cfg.dataset.batch_size,
        cfg.dataset.train_fraction,
        cfg.dataset.validation_fraction,
        cfg.dataset.test_fraction,
        cfg.random_seed,
    )

    # The total number of clients created produced from sampling differs (on different random seeds)
    total_n_clients = len(trainloaders)

    client_fnc = partial(
        create_client,
        trainloaders=trainloaders,
        valloaders=valloaders,
        testloaders=testloaders,
        device=device,
        num_epochs=cfg.training.epochs_per_round,
        learning_rate=cfg.training.learning_rate,
        # There exist other variants of the NIST dataset with different # of classes
        num_classes=cfg.dataset.num_classes,
        num_batches=cfg.training.batches_per_round,
    )
    flwr_strategy: Union[Type[FedAvg], Type[FedAvgSameClients]]
    if cfg.training.same_train_test_clients:
        #  Assign reference to a class
        flwr_strategy = FedAvgSameClients
    else:
        flwr_strategy = FedAvg

    strategy = flwr_strategy(
        min_available_clients=total_n_clients,
        # min number of clients to sample from for fit and evaluate
        # Keep fraction fit low (not zero for consistency reasons with fraction_evaluate)
        # and determine number of clients by the min_fit_clients
        # (it's max of 1. fraction_fit * available clients 2. min_fit_clients)
        fraction_fit=0.001,
        min_fit_clients=cfg.training.num_clients_per_round,
        fraction_evaluate=0.001,
        min_evaluate_clients=cfg.training.num_clients_per_round,
        # evaluate_fn=None, #  Leave empty since it's responsible for the centralized evaluation
        fit_metrics_aggregation_fn=weighted_average,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    client_resources = None
    if device.type == "cuda":
        client_resources = {"num_gpus": 1.0}

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fnc,  # type: ignore
        num_clients=total_n_clients,  # total number of clients in a simulation
        config=fl.server.ServerConfig(num_rounds=cfg.training.num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )

    # Save the results
    results_dir_path = pathlib.Path(cfg.training.results_dir_path)
    if not results_dir_path.exists():
        results_dir_path.mkdir(parents=True)

    distributed_history_dict = {}
    for metric, round_value_tuple_list in history.metrics_distributed.items():
        distributed_history_dict["distributed_test_" + metric] = [
            val for _, val in round_value_tuple_list
        ]
    for metric, round_value_tuple_list in history.metrics_distributed_fit.items():  # type: ignore
        distributed_history_dict["distributed_" + metric] = [
            val for _, val in round_value_tuple_list
        ]
    distributed_history_dict["distributed_test_loss"] = [
        val for _, val in history.losses_distributed
    ]

    results_df = pd.DataFrame.from_dict(distributed_history_dict)
    results_df.to_csv(results_dir_path / "history.csv")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()
