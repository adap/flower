import math
from collections import OrderedDict
from typing import Callable, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision.transforms as transforms
from opacus.dp_model_inspector import DPModelInspector
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from dp_cifar_main import DEVICE, PARAMS, DPCifarClient, Net, test

# Adapted from the PyTorch quickstart and ray simulation (quickstart and extended) examples.


# Define parameters.
NUM_CLIENTS = 2


def client_fn(cid: str) -> fl.client.Client:
    # Load model.
    model = Net()
    # Check model is compatible with Opacus.
    # inspector = DPModelInspector()
    # print(f"Is the model valid? {inspector.validate(model)}")

    # Load data partition (divide CIFAR10 into NUM_CLIENTS distinct partitions, using 30% for validation).
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    data = CIFAR10("./data", train=True, download=True, transform=transform)

    partitions = tuple([len(data) // NUM_CLIENTS for i in range(NUM_CLIENTS)])
    partitioned_data = torch.utils.data.random_split(
        data, partitions, generator=torch.Generator().manual_seed(2)
    )
    client_data = partitioned_data[int(cid)]
    split = math.floor(len(client_data) * PARAMS["train_split"])
    client_trainset = torch.utils.data.Subset(client_data, list(range(0, split)))
    client_testset = torch.utils.data.Subset(
        client_data, list(range(split, len(client_data)))
    )
    client_trainloader = DataLoader(client_trainset, PARAMS["batch_size"])
    client_testloader = DataLoader(client_testset, PARAMS["batch_size"])

    sample_rate = PARAMS["batch_size"] / len(client_trainset)

    return DPCifarClient(model, client_trainloader, client_testloader, sample_rate)


# Define an evaluation function for centralized evaluation (using whole CIFAR10 testset).
def get_evaluate_fn() -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        testset = CIFAR10(root="./data", train=False, transform=transform)
        model = Net()
        # Set weights in model.
        state_dict = OrderedDict(
            {
                k: torch.tensor(np.atleast_1d(v))
                for k, v in zip(model.state_dict().keys(), weights)
            }
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(DEVICE)
        testloader = torch.utils.data.DataLoader(testset, PARAMS["batch_size"])
        loss, accuracy = test(model, testloader)
        # Return metrics.
        return loss, {"accuracy": accuracy}

    return evaluate


def main() -> None:
    # Start Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 1},
        num_rounds=3,
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1, fraction_evaluate=0.1, evaluate_fn=get_evaluate_fn()
        ),
    )


if __name__ == "__main__":
    main()
