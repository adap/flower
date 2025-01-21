"""floco: A Flower Baseline."""
import os
from typing import List, Dict, Tuple, Optional

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner

from flwr.common import Context, NDArrays, Scalar, ndarrays_to_parameters, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from floco.model import SimplexModel, get_weights, set_weights, test
from floco.strategy import Floco

NUM_PARTITIONS = 100
BATCH_SIZE = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def load_datasets(partition_id: int, num_partitions: int):
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions, partition_by="label",
        alpha=0.5, min_partition_size=100,
        self_balancing=True)
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": partitioner})
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=BATCH_SIZE, shuffle=True
    )
    valloader = DataLoader(partition_train_test["test"], batch_size=BATCH_SIZE)
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    
    return trainloader, valloader, testloader

# Define metric aggregation function
def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}

# The `evaluate` function will be called by Flower after every round
def server_evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar],
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
    server_model = SimplexModel(
        endpoints=3, seed=0).to(DEVICE)
    set_weights(server_model, parameters)  # Update model with the latest parameters
    server_model.training = False
    server_model.subregion_parameters = (config["center"], config["radius"])
    _, _, testloader = load_datasets(0, NUM_PARTITIONS)
    loss, accuracy = test(server_model, testloader, DEVICE)
    return loss, {"centralized_accuracy": accuracy}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    tau = context.run_config["tau"]
    rho = context.run_config["rho"]
    endpoints = context.run_config["endpoints"]
    pers_epoch = context.run_config["pers-epoch"]
    num_clients = context.run_config["num-clients"]
    seed = context.run_config["seed"]

    # Initialize model parameters
    initial_model = SimplexModel(endpoints=endpoints, seed=seed)
    ndarrays = get_weights(initial_model)
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = Floco(
        fraction_fit=float(fraction_fit),
        fraction_evaluate=1.0,
        min_available_clients=2,
        tau=tau,
        rho=rho,
        endpoints=endpoints,
        num_clients=num_clients,
        pers_epoch=pers_epoch,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        run_config=context.run_config,
        use_wandb=context.run_config["use-wandb"],
        evaluate_fn=server_evaluate,
    )
    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)