"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

from copy import deepcopy
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class StackedLSTM(nn.Module):
    """StackedLSTM architecture.

    As described in Fei Chen 2018 paper :

    [FedMeta: Federated Meta-Learning with Fast Convergence and Efficient Communication]
    (https://arxiv.org/abs/1802.07876)
    """

    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fully_ = nn.Linear(256, 80)

    def forward(self, text):
        """Forward pass of the StackedLSTM.

        Parameters
        ----------
        text : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        embedded = self.embedding(text)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embedded)
        final_output = self.fully_(lstm_out[:, -1, :])
        return final_output


class FemnistNetwork(nn.Module):
    """Convolutional Neural Network architecture.

    As described in Fei Chen 2018 paper :

    [FedMeta: Federated Meta-Learning with Fast Convergence and Efficient Communication]
    (https://arxiv.org/abs/1802.07876)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = nn.Linear(7 * 7 * 64, 2048)
        self.linear2 = nn.Linear(2048, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu((self.linear1(x)))
        x = self.linear2(x)
        return x


# pylint: disable=too-many-arguments
def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> Tuple[float]:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the optimizer.

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    loss
        The Loss that bas been trained for one epoch
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    net.train()
    for _ in range(epochs):
        net, loss = _train_one_epoch(net, trainloader, device, criterion, optimizer)
    return loss


def _train_one_epoch(
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
    total_loss
        The Loss that has been trained for one epoch.
    """
    total_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        total_loss += loss.item() * labels.size(0)
        loss.backward()
        optimizer.step()
    total_loss = total_loss / len(trainloader.dataset)
    return net, total_loss


# pylint: disable=too-many-locals
def test(
    net: nn.Module,
    trainloader: DataLoader,
    testloader: DataLoader,
    device: torch.device,
    algo: str,
    data: str,
    learning_rate: float,
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    trainloader: DataLoader,
        The DataLoader containing the data to train the network on.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.
    algo: str
        The Algorithm of Federated Learning
    data: str
        The training data type of Federated Learning
    learning_rate: float
        The learning rate for the optimizer.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    if algo == "fedavg_meta":
        optimizer = torch.optim.Adam(
            net.parameters(), lr=learning_rate, weight_decay=0.001
        )
        net.train()
        optimizer.zero_grad()
        if data == "femnist":
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                loss = criterion(net(images), labels)
                loss.backward()
                total_loss += loss * labels.size(0)
            total_loss = total_loss / len(trainloader.dataset)
            optimizer.step()

        elif data == "shakespeare":
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                loss = criterion(net(images), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def train_meta(
    net: nn.Module,
    supportloader: DataLoader,
    queryloader: DataLoader,
    alpha: torch.nn.ParameterList,
    device: torch.device,
    gradient_step: int,
) -> Tuple[float, List]:
    """Train the network on the training set.

    Parameters
    ----------
     net : nn.Module
         The neural network to train.
     supportloader : DataLoader
         The DataLoader containing the data to inner loop train the network on.
     queryloader : DataLoader
         The DataLoader containing the data to outer loop train the network on.
    alpha : torch.nn.ParameterList
         The learning rate for the optimizer.
     device : torch.device
         The device on which the model should be trained, either 'cpu' or 'cuda'.
     gradient_step : int
         The number of inner loop learning

    Returns
    -------
     total_loss
         The Loss that has been trained for one epoch.
     grads
         The gradients that has been trained for one epoch.
    """
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(1):
        loss, grads = _train_meta_one_epoch(
            net, supportloader, queryloader, alpha, criterion, device, gradient_step
        )
    return loss, grads


# pylint: disable=too-many-locals
def _train_meta_one_epoch(
    net: nn.Module,
    supportloader: DataLoader,
    queryloader: DataLoader,
    alpha: torch.nn.ParameterList,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
    gradient_step: int,
) -> Tuple[float, List]:
    """Train for one epoch.

    Parameters
    ----------
     net : nn.Module
         The neural network to train.
     supportloader : DataLoader
         The DataLoader containing the data to inner loop train the network on.
     queryloader : DataLoader
         The DataLoader containing the data to outer loop train the network on.
    alpha : torch.nn.ParameterList
         The learning rate for the optimizer.
     criterion : torch.nn.CrossEntropyLoss
         The loss function to use for training
     device : torch.device
         The device on which the model should be trained, either 'cpu' or 'cuda'.
     gradient_step : int
         The number of inner loop learning

    Returns
    -------
     total_loss
         The Loss that has been trained for one epoch.
     grads
         The gradients that has been trained for one epoch.
    """
    num_adaptation_steps = gradient_step
    train_net = deepcopy(net)
    alpha = [alpha.to(device) for alpha in alpha]
    train_net.train()
    for _ in range(num_adaptation_steps):
        loss_sum = 0.0
        sup_num_sample = []
        sup_total_loss = []
        for images, labels in supportloader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(train_net(images), labels)
            loss_sum += loss * labels.size(0)
            sup_num_sample.append(labels.size(0))
            sup_total_loss.append(loss * labels.size(0))
            grads = torch.autograd.grad(
                loss, list(train_net.parameters()), create_graph=True, retain_graph=True
            )

            for param, grad_, alphas in zip(train_net.parameters(), grads, alpha):
                param.data = param.data - alphas * grad_

            for param in train_net.parameters():
                if param.grad is not None:
                    param.grad.zero_()

    qry_total_loss = []
    qry_num_sample = []
    loss_sum = 0.0
    for images, labels in queryloader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(train_net(images), labels)
        loss_sum += loss * labels.size(0)
        qry_num_sample.append(labels.size(0))
        qry_total_loss.append(loss.item())
    loss_sum = loss_sum / sum(qry_num_sample)
    grads = torch.autograd.grad(loss_sum, list(train_net.parameters()))

    for param in train_net.parameters():
        if param.grad is not None:
            param.grad.zero_()

    grads = [grad_.cpu().numpy() for grad_ in grads]
    loss = sum(sup_total_loss) / sum(sup_num_sample)
    return loss, grads


def test_meta(
    net: nn.Module,
    supportloader: DataLoader,
    queryloader: DataLoader,
    alpha: torch.nn.ParameterList,
    device: torch.device,
    gradient_step: int,
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    supportloader : DataLoader
        The DataLoader containing the data to test the network on.
    queryloader : DataLoader
        The DataLoader containing the data to test the network on.
    alpha : torch.nn.ParameterList
        The learning rate for the optimizer.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.
    gradient_step : int
        The number of inner loop learning

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    test_net = deepcopy(net)
    num_adaptation_steps = gradient_step
    alpha = [alpha_tensor.to(device) for alpha_tensor in alpha]
    test_net.train()
    for _ in range(num_adaptation_steps):
        loss_sum = 0.0
        sup_num_sample = []
        sup_total_loss = []
        for images, labels in supportloader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(test_net(images), labels)
            loss_sum += loss * labels.size(0)
            sup_num_sample.append(labels.size(0))
            sup_total_loss.append(loss)
            grads = torch.autograd.grad(
                loss, list(test_net.parameters()), create_graph=True, retain_graph=True
            )

            for param, grad_, alphas in zip(test_net.parameters(), grads, alpha):
                param.data -= alphas * grad_

            for param in test_net.parameters():
                if param.grad is not None:
                    param.grad.zero_()

    test_net.eval()
    correct, total, loss = 0, 0, 0.0
    for images, labels in queryloader:
        images, labels = images.to(device), labels.to(device)
        outputs = test_net(images)
        loss += criterion(outputs, labels).item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    if len(queryloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss = loss / total
    accuracy = correct / total
    return loss, accuracy
