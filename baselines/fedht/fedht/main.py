"""Run main for fedht baseline."""

import pickle
import random
import torch

import flwr as fl
import hydra
import numpy as np
from flwr.common import NDArrays, ndarrays_to_parameters
from flwr.server.strategy.strategy import Strategy
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedht.client import generate_client_fn_mnist, generate_client_fn_simII
from fedht.fedht import FedHT
from fedht.server import fit_round, gen_evaluate_fn
from fedht.utils import MyDataset, sim_data


@hydra.main(config_path="conf", config_name="base_mnist", version_base=None)
def main(cfg: DictConfig):
    """Run main file for fedht baseline.

    Parameters
    ----------
    cfg : DictConfig
        Config file for federated baseline; read from fedht/conf.
    """

    # set device to cuda:0, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        num_features = cfg.num_features
        num_classes = cfg.num_classes
        dataset = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
        test_dataset = dataset.load_split("test").with_format("numpy")
        testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        # set client function
        client_fn = generate_client_fn_mnist(
            dataset,
            cfg=cfg
        )

    elif cfg.data == "simII":

        # set seed
        np.random.seed(2024)

        # simulate data from Simulation II in Tong et al
        num_obs = cfg.num_obs
        num_clients = cfg.num_clients
        num_features = cfg.num_features
        num_classes = cfg.num_classes

        dataset = sim_data(num_obs, num_clients, num_features, 1, 1)
        X_test, y_test = sim_data(num_obs, 1, num_features, 1, 1)
        test_dataset = MyDataset(X_test[0, :, :], y_test[:, 0])
        testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        # set client function
        client_fn = generate_client_fn_simII(
            dataset,
            cfg=cfg
        )

    # initialize global model to all zeros
    weights = np.zeros((num_classes, num_features))
    bias = np.zeros(num_classes)
    init_params_arr: NDArrays = [weights, bias]
    init_params = ndarrays_to_parameters(init_params_arr.copy())

    strategy: Strategy
    if cfg.agg == "fedht":
        # define strategy: fedht
        strategy = FedHT(
            min_available_clients=cfg.strategy.min_available_clients,
            num_keep=cfg.num_keep,
            evaluate_fn=gen_evaluate_fn(testloader, cfg, device),
            on_fit_config_fn=fit_round,
            iterht=cfg.iterht,
            initial_parameters=init_params,
        )
    elif cfg.agg == "fedavg":
        # define strategy: fedavg
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=cfg.strategy.min_available_clients,
            evaluate_fn=gen_evaluate_fn(testloader, cfg, device),
            on_fit_config_fn=fit_round,
            initial_parameters=init_params,
        )
    else:
        raise ValueError("Must select either fedht or fedavg for the aggregation strategy in this baseline.")

    # start simulation
    np.random.seed(2025)
    hist_mnist = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={
            "num_cpus": cfg.client_resources.num_cpus,
            "num_gpus": cfg.client_resources.num_gpus,
        },
    )

    if cfg.iterht:
        iterstr = "iter"
    else:
        iterstr = ""

    filename = (
        cfg.data
        + "_"
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
