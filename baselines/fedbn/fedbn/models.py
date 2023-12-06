"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

from typing import Tuple

import torch
from torch import nn


# pylint: disable=too-many-instance-attributes
class CNNModel(nn.Module):
    """CNN model proposed in the FedBN paper."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        """Forward pass."""
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = nn.functional.max_pool2d(x, 2)

        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = nn.functional.max_pool2d(x, 2)

        x = nn.functional.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = nn.functional.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        x = nn.functional.relu(x)

        x = self.fc3(x)
        return x


# pylint: disable=too-many-locals
def train(model, traindata, epochs, l_r, device) -> Tuple[float, float]:
    """Train the network."""
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=l_r)

    # Train the network
    model.to(device)
    model.train()
    total_loss = 0.0
    for _ in range(epochs):  # loop over the dataset multiple times
        total = 0.0
        correct = 0
        for _i, data in enumerate(traindata, 0):
            images, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return total_loss / len(traindata), correct / total


def test(model, testdata, device) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    # Define loss and metrics
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0

    # Evaluate the network
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data in testdata:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    loss = loss / len(testdata)
    return loss, accuracy
