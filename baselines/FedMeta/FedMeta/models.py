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
from copy import deepcopy


class StackedLSTM(nn.Module):
    def __init__(self):
        super(StackedLSTM, self).__init__()

        self.embedding = nn.Embedding(80, 8)
        self.lstm = nn.LSTM(8, 256, num_layers=2, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(256, 80)

    def forward(self, text):
        embedded = self.embedding(text)
        self.lstm.flatten_parameters()
        lstm_out, _ = self.lstm(embedded)
        final_output = self.fc(lstm_out[:, -1, :])
        return final_output


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



    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu((self.linear1(x)))
        x = self.linear2(x)
        return x


def train(  # pylint: disable=too-many-arguments
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
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        net, loss = _train_one_epoch(
            net, trainloader, device, criterion, optimizer
        )
    return loss


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
    total_loss = 0.0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        # loss = criterion(net(images.to(torch.float32)), labels)
        loss = criterion(net(images), labels)
        total_loss += loss.item() * labels.size(0)
        loss.backward()
        optimizer.step()
    total_loss = total_loss / len(trainloader.dataset)
    return net, total_loss


def test(
        net: nn.Module,
        trainloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        learning_rate: float
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
    total_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=0.001)
    net.train()
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        # loss = criterion(net(images.to(torch.float32)), labels)
        loss = criterion(net(images), labels)
        total_loss += loss * labels.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # total_loss = total_loss / len(trainloader.dataset)
    # optimizer.zero_grad()
    # total_loss.backward()
    # optimizer.step()

    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            # outputs = net(images.to(torch.float32))
            outputs = net(images)
            loss += criterion(outputs, labels).item() * labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy, total
#
#
def train_meta(  # pylint: disable=too-many-arguments
        net: nn.Module,
        supportloader: DataLoader,
        queryloader: DataLoader,
        alpha,
        device: torch.device,
        learning_rate: float,
) -> Tuple[float]:
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
    for _ in range(1):
        loss, grads = _train_meta_one_epoch(
            net, supportloader, queryloader, alpha, criterion, learning_rate, device
        )
    return loss, grads


def _train_meta_one_epoch(  # pylint: disable=too-many-arguments
        net: nn.Module,
        supportloader: DataLoader,
        queryloader: DataLoader,
        alpha,
        criterion: torch.nn.CrossEntropyLoss,
        learning_rate: float,
        device: torch.device,
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
    num_adaptation_steps = 1
    all_adaptation_losses = []
    train_net = deepcopy(net)
    # alpha = [alpha.to(device) for alpha in alpha]
    for step in range(num_adaptation_steps):
        loss_sum = 0.0
        sup_num_sample = []
        sup_total_loss = []
        for images, labels in supportloader:
            images, labels = images.to(device), labels.to(device)
            # loss = criterion(train_net(images.to(torch.float32)), labels)
            loss = criterion(train_net(images), labels)
            loss_sum += loss * labels.size(0)
            sup_num_sample.append(labels.size(0))
            sup_total_loss.append(loss * labels.size(0))
            grads = torch.autograd.grad(loss, list(train_net.parameters()), create_graph=True, retain_graph=True)

            for p, g in zip(train_net.parameters(), grads):
                p.data.add_(g.data, alpha=-learning_rate)

            for p in train_net.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # for p, g, a in zip(train_net.parameters(), grads, alpha):
            #     p.data = p.data - a * g
            #
            # for p in train_net.parameters():
            #     if p.grad is not None:
            #         p.grad.zero_()

    qry_total_loss = []
    qry_num_sample = []
    loss_sum = 0.0
    for images, labels in queryloader:
        images, labels = images.to(device), labels.to(device)
        # loss = criterion(train_net(images.to(torch.float32)), labels)
        loss = criterion(train_net(images), labels)
        loss_sum += loss * labels.size(0)
        qry_num_sample.append(labels.size(0))
        qry_total_loss.append(loss.item())
    loss_sum = loss_sum / sum(qry_num_sample)
    grads = torch.autograd.grad(loss_sum, list(train_net.parameters()))

    for p in train_net.parameters():
        if p.grad is not None:
            p.grad.zero_()

    grads = [g.cpu().numpy() for g in grads]
    average_adaptation_loss = sum(sup_total_loss) / sum(sup_num_sample)
    return average_adaptation_loss, grads


def test_meta(
        net: nn.Module,
        supportloader: DataLoader,
        queryloader: DataLoader,
        alpha,
        device: torch.device,
        learning_rate: float,
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
    test_net = deepcopy(net)
    num_adaptation_steps = 1
    alpha = [alpha_tensor.to(device) for alpha_tensor in alpha]
    test_net.train()
    for step in range(num_adaptation_steps):
        loss_sum = 0.0
        sup_num_sample = []
        sup_total_loss = []
        for images, labels in supportloader:
            images, labels = images.to(device), labels.to(device)
            # loss = criterion(test_net(images.to(torch.float32)), labels)
            loss = criterion(test_net(images), labels)
            loss_sum += loss * labels.size(0)
            sup_num_sample.append(labels.size(0))
            sup_total_loss.append(loss)
            grads = torch.autograd.grad(loss, list(test_net.parameters()), create_graph=True, retain_graph=True)

            for p, g in zip(test_net.parameters(), grads):
                p.data.add_(g.data, alpha=-learning_rate)

            for p in test_net.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            # for p, g, a in zip(test_net.parameters(), grads, alpha):
            #     p.data -= a * g
            #
            # for p in test_net.parameters():
            #     if p.grad is not None:
            #         p.grad.zero_()

    test_net.eval()
    correct, total, loss = 0, 0, 0.0
    for images, labels in queryloader:
        images, labels = images.to(device), labels.to(device)
        # outputs = test_net(images.to(torch.float32))
        outputs = test_net(images)
        loss += criterion(outputs, labels).item() * labels.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    if len(queryloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss = loss / total
    accuracy = correct / total
    return loss, accuracy, total



