import math
from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from flwr.common.typing import Scalar

from dp_cifar_main import DEVICE, PARAMS, DPCifarClient, Net, test

# Adapted from the PyTorch quickstart and ray simulation (quickstart and extended) examples.


# Define parameters.
NUM_CLIENTS = 2


def client_fn(cid: str) -> fl.client.Client:
    # Load model.
    model = Net()
    # Check model is compatible with Opacus.

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

    return DPCifarClient(model, client_trainloader, client_testloader).to_client()


# Define an evaluation function for centralized evaluation (using whole CIFAR10 testset).
def get_evaluate_fn() -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ):
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
                for k, v in zip(model.state_dict().keys(), parameters)
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
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1, fraction_evaluate=0.1, evaluate_fn=get_evaluate_fn()
        ),
    )


if __name__ == "__main__":
    main()
