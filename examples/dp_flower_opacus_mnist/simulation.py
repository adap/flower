import multiprocessing as mp
import sys
from collections import OrderedDict
from functools import partial
from pkgutil import get_data
from typing import Optional, Tuple

import numpy as np
import torch
from loguru import logger
from opacus import PrivacyEngine
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10

sys.path.insert(0, "../../src/py")

import flwr as fl
from flwr.client import DPClient, test
from flwr.dataset.utils.common import create_partitioned_dataset

print(fl.__file__)


class Net(nn.Module):
    """An example PyTorch convolutional neural network."""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.gn1 = nn.GroupNorm(int(6 / 3), 6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(int(16 / 4), 16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Neural network forward pass."""
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.gn1(x)
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.gn2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def get_dataloaders(batch_size: int, num_clients: int, cid: int):
    """Function for the client to get train and test dataloaders."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    test_dataset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    train_sampler = DistributedSampler(
        train_dataset, num_clients, cid, shuffle=True, drop_last=True
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, batch_sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader


def load_data(batch_size: int, num_clients: int, cid: int):
    """Function for the client to get train and test dataloaders."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    train_dataset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    test_dataset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset)


def get_client_fn(
    num_clients: int,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float,
    batch_size: int,
    learning_rate: float,
    device: str,
):
    """Client function."""

    def client_fn(cid: str):
        module = Net()
        optimizer = optim.SGD(module.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        privacy_engine = PrivacyEngine()
        # train_loader, test_loader = get_dataloaders(batch_size, num_clients, int(cid))
        train_loader, test_loader = load_data(batch_size, num_clients, int(cid))
        client = DPClient(
            module=module,
            optimizer=optimizer,
            criterion=criterion,
            privacy_engine=privacy_engine,
            train_loader=train_loader,
            test_loader=test_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            device=device,
        )
        return client

    return client_fn


def get_eval_fn(batch_size: int, device: str):
    """Get the evaluation function for scoring on the server."""

    def eval_fn(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Server-side evaluation function."""
        model = Net()
        criterion = nn.CrossEntropyLoss()

        def accuracy(predictions, actuals):
            total = actuals.size(0)
            correct = (predictions.eq(actuals)).sum().item()
            return correct / total

        # Set weights in model.
        state_dict = OrderedDict(
            {k: torch.tensor(np.atleast_1d(v)) for k, v in zip(model.state_dict().keys(), weights)}
        )
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        # testloader = get_dataloaders(batch_size, 1, 0)[1]
        _, test_loader = load_data(batch_size, 1, 0)
        loss, _, accuracy = test(model, criterion, test_loader, device, accuracy=accuracy)
        # Return metrics.
        return loss, {"accuracy": accuracy}

    return eval_fn


def simulation_main(
    num_clients: int,
    num_rounds: int,
    fit_clients: int,
    available_clients: int,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float,
    batch_size: int,
    learning_rate: float,
    client_device: str,
    server_device: str,
):
    """Run the simulation."""
    fraction_clients = fit_clients / available_clients
    fl.simulation.start_simulation(
        client_fn=get_client_fn(
            num_clients=num_clients,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=client_device,
        ),
        num_clients=num_clients,
        client_resources={"num_cpus": 1},
        num_rounds=num_rounds,
        strategy=fl.server.strategy.FedAvgDp(
            fraction_fit=fraction_clients,
            fraction_eval=fraction_clients,
            min_fit_clients=fit_clients,
            min_available_clients=available_clients,
            eval_fn=get_eval_fn(batch_size, server_device),
        ),
    )


def start_client(
    num_clients: int,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float,
    batch_size: int,
    learning_rate: float,
    device: str,
    cid: int,
):
    client_fn = get_client_fn(
        num_clients=num_clients,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        epochs=epochs,
        max_grad_norm=max_grad_norm,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device=device,
    )
    logger.info("Starting client {}", cid)
    return fl.client.start_numpy_client("[::]:8080", client=client_fn(cid))


def main(
    num_clients: int,
    num_rounds: int,
    fit_clients: int,
    available_clients: int,
    target_epsilon: float,
    target_delta: float,
    epochs: int,
    max_grad_norm: float,
    batch_size: int,
    learning_rate: float,
    client_device: str,
    server_device: str,
):
    """Main."""
    fraction_clients = fit_clients / available_clients
    strategy = fl.server.strategy.FedAvgDp(
        fraction_fit=fraction_clients,
        fraction_eval=fraction_clients,
        min_fit_clients=fit_clients,
        min_available_clients=available_clients,
        eval_fn=get_eval_fn(batch_size, server_device),
    )
    server_process = mp.Process(
        target=fl.server.start_server,
        args=["[::]:8080"],
        kwargs=dict(config={"num_rounds": num_rounds}, strategy=strategy),
    )
    server_process.start()
    with mp.Pool(num_clients) as pool:
        start = partial(
            start_client,
            num_clients,
            target_epsilon,
            target_delta,
            epochs,
            max_grad_norm,
            batch_size,
            learning_rate,
            client_device,
        )

        pool.map(start, range(num_clients))
    server_process.kill()


def main_test():
    batch_size = 32
    num_clients = 1
    cid = 0
    module = Net()
    optimizer = optim.SGD(module.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    privacy_engine = PrivacyEngine()
    # train_loader, test_loader = get_dataloaders(batch_size, num_clients, int(cid))
    train_loader, test_loader = load_data(batch_size, num_clients, int(cid))

    def accuracy(predictions, actuals):
        total = actuals.size(0)
        correct = (predictions.eq(actuals)).sum().item()
        return correct / total

    test(module, criterion, test_loader, "cpu", accuracy=accuracy)


if __name__ == "__main__":
    NUM_CLIENTS = 1
    AVAILABLE_CLIENTS = 4
    FIT_CLIENTS = 3
    NUM_ROUNDS = 1
    TARGET_EPSILON = 1.0
    TARGET_DELTA = 0.1
    MAX_GRAD_NORM = 1.0
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    EPOCHS = 2
    CLIENT_DEVICE = "cpu"
    SERVER_DEVICE = "cpu"
    # main(
    #     num_clients=NUM_CLIENTS,
    #     num_rounds=NUM_ROUNDS,
    #     fit_clients=FIT_CLIENTS,
    #     available_clients=AVAILABLE_CLIENTS,
    #     target_epsilon=TARGET_EPSILON,
    #     target_delta=TARGET_DELTA,
    #     epochs=EPOCHS,
    #     max_grad_norm=MAX_GRAD_NORM,
    #     batch_size=BATCH_SIZE,
    #     learning_rate=LEARNING_RATE,
    #     client_device=CLIENT_DEVICE,
    #     server_device=SERVER_DEVICE,
    # )
    main_test()
