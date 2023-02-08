from collections import OrderedDict
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from numpy import copy
from torch.nn import Conv2d, Linear, MaxPool2d, Module
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from flwr.common.typing import Config, Metrics, NDArrays


# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
# borrowed from Pytorch quickstart example
class Net(Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = Conv2d(3, 6, 5)
        self.pool = MaxPool2d(2, 2)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# adapted from Pytorch quickstart example
def train(
    net: Module,
    dataloader: DataLoader,
    device: torch.device,
    epochs: int = 1,
) -> float:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    correct, total, loss = 0, 0, 0.0
    net.train()
    last_loss = 0.0
    for _ in range(epochs):
        last_loss = 0.0
        for idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            last_loss += loss.item()

    return float(last_loss)


# borrowed from Pytorch quickstart example
def test(net: torch.nn.Module, dataloader, device: torch.device) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def model_to_arrays(model: torch.nn.Module) -> NDArrays:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.Module, params: NDArrays):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def fit_config(server_round: int) -> Config:
    """Return a configuration with static batch size and (local) epochs."""
    config: Config = {
        "epochs": 5,  # number of local epochs
        "batch_size": 32,
    }
    return config


def evaluate_config(server_round: int) -> Config:
    """Return a configuration with static batch size and (local) epochs."""
    config: Config = {
        "epochs": 5,  # number of local epochs
        "batch_size": 32,
    }
    return config


def get_evaluate_fn(
    testset: CIFAR10,
) -> Callable[[int, NDArrays, Config], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: NDArrays, config: Config
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0")

        model = Net()
        set_params(model, parameters)

        model.to(device)

        testloader = DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, accuracy

    return evaluate
