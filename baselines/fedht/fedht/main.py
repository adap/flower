"""
Author: Chance Johnstone
Purpose: main function for fedht baseline Simulation II example from Tong et al 2020

Notes
-----
Code included in this baseline generated with the help of numerous 
Flower, Python, and PyTorch resources.
"""

import pickle
import random

import flwr as fl
import hydra
import numpy as np
from flwr.common import NDArrays, ndarrays_to_parameters
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedht.client import generate_client_fn_mnist, generate_client_fn_simII
from fedht.fedht import FedHT
from fedht.model import LogisticRegression
from fedht.server import fit_round, get_evaluate_fn
from fedht.utils import MyDataset, sim_data


@hydra.main(config_path="conf", config_name="base_mnist", version_base=None)
def main(cfg: DictConfig):

    # set seed
    random.seed(2024)

    if cfg.data == "mnist":
        # set up mnist data
        # set partitioner
        partitioner = PathologicalPartitioner(
            num_partitions=cfg.num_clients,
            partition_by="label",
            num_classes_per_partition=2,
            class_assignment_mode="first-deterministic",
        )

        # load MNIST data
        num_features = 28 * 28
        num_classes = 10
        dataset = FederatedDataset(dataset="mnist", partitioners={"train": num_classes})
        test_dataset = dataset.load_split("test").with_format("numpy")
        testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        # define model
        model = LogisticRegression(num_features, num_classes)

        # set client function
        client_fn = generate_client_fn_mnist(
            dataset,
            num_features=num_features,
            num_classes=num_classes,
            model=model,
            cfg=cfg,
        )

    elif cfg.data == "simII":

        # simulate data from Simulation II in Tong et al
        num_obs = 1000
        num_clients = cfg.num_clients
        num_features = 1000
        num_classes = 2

        dataset = sim_data(num_obs, num_clients, num_features, 1, 1)
        X_test, y_test = sim_data(num_obs, 1, num_features, 1, 1)
        test_dataset = MyDataset(X_test[0, :, :], y_test[:, 0])
        testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        # define model
        model = LogisticRegression(num_features, num_classes)

        # set client function
        client_fn = generate_client_fn_simII(
            dataset,
            num_features=num_features,
            num_classes=num_classes,
            model=model,
            cfg=cfg,
        )

    # initialize global model to all zeros
    weights = np.zeros((num_classes, num_features))
    bias = np.zeros(num_classes)
    init_params: NDArrays = (weights, bias)
    init_params = ndarrays_to_parameters(init_params)

    # define strategy: fedht
    strategy_fedht = FedHT(
        min_available_clients=cfg.strategy.min_available_clients,
        num_keep=cfg.num_keep,
        evaluate_fn=get_evaluate_fn(testloader, model),
        on_fit_config_fn=fit_round,
        iterht=cfg.iterht,
        initial_parameters=init_params,
    )

    # define strategy: fedavg
    strategy_fedavg = fl.server.strategy.FedAvg(
        min_available_clients=cfg.strategy.min_available_clients,
        evaluate_fn=get_evaluate_fn(testloader, model),
        on_fit_config_fn=fit_round,
        initial_parameters=init_params,
    )

    if cfg.agg == "fedht":
        strategy = strategy_fedht
    elif cfg.agg == "fedavg":
        strategy = strategy_fedavg
    else:
        print("Must select either fedht or fedavg for the aggregation strategy.")

    # # start simulation
    random.seed(2025)
    hist_mnist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": cfg.client_resources.num_cpus},
    )

    if cfg.iterht == True:
        iterstr = "iter"
    else:
        iterstr = ""

    filename = (
        "simII_"
        + cfg.agg
        + iterstr
        + "_local"
        + str(cfg.num_local_epochs)
        + "_lr"
        + str(cfg.learning_rate)
        + "_numkeep"
        + str(cfg.num_keep)
        + ".pkl"
    )

    with open(filename, "wb") as file:
        pickle.dump(hist_mnist, file)


if __name__ == "__main__":
    main()
