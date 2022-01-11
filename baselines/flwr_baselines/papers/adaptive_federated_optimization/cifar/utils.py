from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from flwr.common.parameter import weights_to_parameters
from flwr.common.typing import Parameters, Scalar, Weights
from flwr.server.history import History
from flwr.dataset.utils.common import XY, create_lda_partitions
from torch.nn import GroupNorm, Module
from torchvision.models import ResNet, resnet18
from typing import Callable, Dict, Optional, Tuple


def partition_and_save(
    dataset: XY,
    fed_dir: Path,
    dirichlet_dist: np.ndarray = None,
    num_partitions: int = 500,
    concentration: float = 0.1,
) -> np.ndarray:
    # Create partitions
    clients_partitions, dist = create_lda_partitions(
        dataset=dataset,
        dirichlet_dist=dirichlet_dist,
        num_partitions=num_partitions,
        concentration=concentration,
    )
    # Save partions
    for idx, partition in enumerate(clients_partitions):
        path_dir = fed_dir / f"{idx}"
        path_dir.mkdir(exist_ok=True, parents=True)
        torch.save(partition, path_dir / "train.pt")

    return dist


def train(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net: Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def gen_cifar10_on_fit_config_fn(
    epochs_per_round: int, batch_size: int, client_learning_rate: float
) -> Callable[[int], Dict[str, Scalar]]:
    def on_fit_config(rnd: int) -> Dict[str, Scalar]:
        """Return a configuration with specific client learning rate."""
        local_config = {
            "epoch_global": rnd,
            "epochs": epochs_per_round,
            "batch_size": batch_size,
            "client_learning_rate": client_learning_rate,
        }
        return local_config

    return on_fit_config


def plot_metric_from_history(
    hist: History,
    metric_str: str,
    strategy_name: str,
    expected_maximum: float,
    save_path: Path,
):
    x, y = zip(*hist.metrics_centralized[metric_str])
    plt.plot(x, y * 100)  # Accuracy 0-100%
    # Set expected graph
    plt.axhline(y=expected_maximum, color="r", linestyle="--")
    plt.title(f"Centralized Validation - {strategy_name}")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.savefig(save_path)
