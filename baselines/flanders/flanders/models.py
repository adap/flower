"""Models for FLANDERS experiments."""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def roc_auc_multiclass(y_true, y_pred):
    """Compute the ROC AUC for multiclass classification."""
    l_b = LabelBinarizer()
    l_b.fit(y_true)
    y_true = l_b.transform(y_true)
    y_pred = l_b.transform(y_pred)
    return roc_auc_score(y_true, y_pred, multi_class="ovr")


# Source:
# github.com/bladesteam/blades/blob/master/src/blades/models/mnist/mlp.py
class MnistNet(nn.Module):
    """Simple MLP for MNIST."""

    def __init__(self):
        """Initialize the model."""
        super().__init__()
        # number of hidden nodes in each layer (512)
        hidden_1 = 128
        hidden_2 = 256
        # linear layer (784 -> hidden_1)
        self.fc1 = nn.Linear(28 * 28, hidden_1)
        # linear layer (n_hidden -> hidden_2)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        # linear layer (n_hidden -> 10)
        self.fc3 = nn.Linear(hidden_2, 10)
        # dropout layer (p=0.2)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        """Forward pass through the network."""
        # flatten image input
        x = x.view(-1, 28 * 28)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add hidden layer, with relu activation function
        x = F.relu(self.fc2(x))
        # add dropout layer
        x = self.dropout(x)
        # add output layer
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def train_mnist(model, dataloader, epochs, device):
    """Train the network on the training set."""
    n_total_steps = len(dataloader)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            # origin shape: [100, 1, 28, 28]
            # resized: [100, 784]
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Step [{i+1}/{n_total_steps}], "
                    f"Loss: {loss.item():.4f}"
                )


# pylint: disable=too-many-locals
def test_mnist(model, dataloader, device):
    """Validate the network on the entire test set."""
    loss = 0
    model.eval()
    criterion = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in dataloader:
            images = images.reshape(-1, 28 * 28).to(device)
            labels = labels.to(device)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            loss += criterion(outputs, labels).item()
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
        y_true = list(itertools.chain(*y_true))
        y_pred = list(itertools.chain(*y_pred))
        auc = roc_auc_multiclass(y_true, y_pred)
        acc = n_correct / n_samples
    return loss, acc, auc


# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
# borrowed from Pytorch quickstart example
class CifarNet(nn.Module):
    """Simple CNN for CIFAR-10."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# borrowed from Pytorch quickstart example
def train_cifar(net, trainloader, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


# borrowed from Pytorch quickstart example
# pylint: disable=too-many-locals
def test_cifar(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    y_true, y_pred = [], []
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            y_true.append(labels.cpu().numpy())
            y_pred.append(predicted.cpu().numpy())
        y_true = list(itertools.chain(*y_true))
        y_pred = list(itertools.chain(*y_pred))
        auc = roc_auc_multiclass(y_true, y_pred)
    accuracy = correct / total
    return loss, accuracy, auc
