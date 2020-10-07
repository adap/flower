# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""PyTorch CIFAR-10 image classification.

The code is generally adapted from 'PyTorch: A 60 Minute Blitz'. Further
explanations are given in the official PyTorch tutorial:

https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""


from collections import OrderedDict
from os import path
from logging import DEBUG
from typing import Optional, Tuple


import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import flwr as fl
from flwr_experimental.baseline.dataset.pytorch_cifar_partitioned import (
    CIFAR10PartitionedDataset,
)
from flwr.common.logger import log

# from flwr_experimental.baseline.model.mobilenetv2_cifar import MobileNetV2 # TODO: fixme

DATA_ROOT = "~/.flower/data/cifar-10"


def load_model(device) -> torch.nn.ModuleList:
    """Load model (ResNet-18)."""
    # return MobileNetV2().to(device) # TODO: fixme
    return torchvision.models.resnet18(num_classes=10).to(device)  # Or: mobilenet_v2


def get_weights(model: torch.nn.ModuleList) -> fl.common.Weights:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


# pylint: disable=unused-argument
def load_data(
    cid: int, root_dir: str = DATA_ROOT, load_testset: bool = False
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load CIFAR-10 (training and test set)."""
    root_dir = path.expanduser(root_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    # Load EC-10 partition
    trainset = CIFAR10PartitionedDataset(
        partition_id=cid, root_dir=root_dir, transform=transform
    )

    testset = None
    if load_testset:
        testset = torchvision.datasets.CIFAR10(
            root=root_dir, train=False, download=True, transform=transform
        )

    return trainset, testset


def train(
    cid: str,
    model: torch.nn.ModuleList,
    trainloader: torch.utils.data.DataLoader,
    epoch_global: int,
    epochs: int,
    device: torch.device,  # pylint: disable=no-member
    batches_per_episode: Optional[int] = None,
) -> None:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

    # Fast-forward scheduler to the right epoch 
    for _ in range(epoch_global):
        scheduler.step()

    log(DEBUG, f"Training {epochs} epoch(s) w/ {len(trainloader)} batches each")
    model.train()
    # Train the network
    for epoch in range(epochs):  # loop over the dataset multiple times
        log(DEBUG, f"Training epoch: {epoch}/{epochs}")
        scheduler.step()
        running_loss = 0.0
        for i, (data, target) in enumerate(trainloader, 0):
            images, labels = data.to(device), target.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2 == 0:  # log every other mini-batch
                log(DEBUG, "[%3d/%3d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

            if batches_per_episode is not None and i >= batches_per_episode:
                break


def test(
    model: torch.nn.ModuleList,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,  # pylint: disable=no-member
) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
