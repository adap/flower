"""fedhomo: A Flower Baseline."""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


class MnistNet(nn.Module):
    """Simple MLP for MNIST (784 → 128 → 64 → 10)."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        """Forward pass."""
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CifarNet(nn.Module):
    """Simple CNN for CIFAR-10 (adapted from 'PyTorch: A 60 Minute Blitz')."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_model(dataset: str) -> nn.Module:
    """Return the model for the given dataset.

    Args:
        dataset: Dataset name, either 'mnist' or 'cifar10'.

    Returns:
        An instance of the appropriate model.
    """
    if dataset == "mnist":
        return MnistNet()
    elif dataset == "cifar10":
        return CifarNet()
    else:
        raise ValueError(f"Unsupported dataset: '{dataset}'. Choose 'mnist' or 'cifar10'.")


def train(net, trainloader, epochs, device):
    """Train the model on the training set.

    Args:
        net: The model to train.
        trainloader: DataLoader for the training set.
        epochs: Number of training epochs.
        device: Device to train on.

    Returns:
        Average training loss over all batches.
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    return running_loss / len(trainloader)


def test(net, testloader, device):
    """Validate the model on the test set.

    Args:
        net: The model to evaluate.
        testloader: DataLoader for the test set.
        device: Device to evaluate on.

    Returns:
        Tuple of (loss, accuracy).
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

    return loss / len(testloader), correct / len(testloader.dataset)


def get_weights(net):
    """Extract model parameters as numpy arrays from state dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
