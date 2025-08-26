"""Run main for fedht baseline."""

import torch
import flwr as fl
import numpy as np
from flwr.server import (
    ServerApp, 
    ServerAppComponents, 
    ServerConfig, 
    SimpleClientManager
)
from flwr.common import NDArrays, ndarrays_to_parameters, Context
from flwr.server.strategy.strategy import Strategy
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader

from fedht.fedht import FedHT
from fedht.server import (
    gen_evaluate_fn, 
    ResultsSaverServer, 
    save_results_and_clean_dir
)

def server_fn(context: Context):
    """Run main file for fedht baseline.

    Parameters
    ----------
    cfg : DictConfig
        Config file for federated baseline; read from fedht/conf.
    """

    # set device to cuda:0, if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if context.run_config["data"] == "mnist":
        # set up mnist data
        # set partitioner
        np.random.seed(context.run_config["seed"])
        partitioner = PathologicalPartitioner(
            num_partitions=context.run_config["num_clients"],
            partition_by="label",
            num_classes_per_partition=2,
            class_assignment_mode="first-deterministic",
        )

        # load MNIST data
        num_features = context.run_config["num_features"]
        num_classes = context.run_config["num_classes"]
        dataset = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
        test_dataset = dataset.load_split("test").with_format("numpy")
        testloader = DataLoader(test_dataset, batch_size=context.run_config["batch_size"], shuffle=False)

    # initialize global model to all zeros
    weights = np.zeros((num_classes, num_features))
    bias = np.zeros(num_classes)
    init_params_arr: NDArrays = [weights, bias]
    init_params = ndarrays_to_parameters(init_params_arr.copy())

    strategy: Strategy
    if context.run_config["agg"] == "fedht":
        # define strategy: fedht
        strategy = FedHT(
            min_available_clients=context.run_config["min_available_clients"],
            num_keep=context.run_config["num_keep"],
            evaluate_fn=gen_evaluate_fn(testloader, context, device),
            iterht=context.run_config["iterht"],
            initial_parameters=init_params,
        )
    elif context.run_config["agg"] == "fedavg":
        # define strategy: fedavg
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=context.run_config["min_available_clients"],
            evaluate_fn=gen_evaluate_fn(testloader, context, device),
            initial_parameters=init_params,
        )
    else:
        raise ValueError("Must select either fedht or fedavg for the aggregation strategy in this baseline.")

    if context.run_config["iterht"]:
        iterstr = "iter"
    else:
        iterstr = ""

    filename = (
        context.run_config["data"]
        + "_"
        + context.run_config["agg"]
        + iterstr
        + "_local"
        + str(context.run_config["num_local_epochs"])
        + "_lr"
        + str(context.run_config["learning_rate"])
        + "_wd"
        + str(context.run_config["weight_decay"])
        + "_numkeep"
        + str(context.run_config["num_keep"])
        + "_fold"
        + str(context.run_config["seed"])
        + ".pkl"
    )

    config = ServerConfig(num_rounds=context.run_config["num_rounds"])
    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        strategy=strategy,
        client_manager=client_manager,
        results_saver_fn=save_results_and_clean_dir,
        context=context,
    )

    return ServerAppComponents(server=server, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
