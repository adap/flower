import argparse
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

sys.path.insert(0, "../../src/py")

import flwr as fl
from flwr.client import DPClient


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


def start_client(cid: int):
    device = "cpu"
    module = Net().to(device)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.001, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    privacy_engine = PrivacyEngine()
    train_loader, test_loader = load_data()
    dp_client = DPClient(
        module=module,
        optimizer=optimizer,
        criterion=criterion,
        privacy_engine=privacy_engine,
        train_loader=train_loader,
        test_loader=test_loader,
        target_epsilon=0.9,
        target_delta=0.1,
        epochs=1,
        max_grad_norm=1.0,
        device=device,
    )
    logger.info("Starting client {}", cid)
    fl.client.start_numpy_client("[::]:8080", client=dp_client)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cid",
        type=int,
        default=0,
        help="Client number.",
    )
    args = parser.parse_args()
    start_client(args.cid)
