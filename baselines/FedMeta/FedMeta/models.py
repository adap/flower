"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Femnist_network(nn.Module):
    """Convolutional Neural Network architecture.

    As described in McMahan 2017 paper :

    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)
    """

    def __init__(self) -> None:
        super(Femnist_network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = nn.Linear(7 * 7 * 64, 2048)
        self.linear2 = nn.Linear(2048, 62)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        input_tensor : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = torch.relu(self.conv1(input_tensor))
        output_tensor = self.maxpool1(output_tensor)
        output_tensor = torch.relu(self.conv2(output_tensor))
        output_tensor = self.maxpool2(output_tensor)
        output_tensor = torch.flatten(output_tensor, start_dim=1)
        output_tensor = torch.relu((self.linear1(output_tensor)))
        output_tensor = self.linear2(output_tensor)
        return output_tensor


def train(  # pylint: disable=too-many-arguments
        net: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        epochs: int,
        learning_rate: float,
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(
            net, trainloader, device, criterion, optimizer
        )


def _train_one_epoch(  # pylint: disable=too-many-arguments
        net: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.Adam,
) -> nn.Module:
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(images.to(torch.float32)), labels)
        loss.backward()
        optimizer.step()
    return net


def test(
        net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images.to(torch.float32))
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
