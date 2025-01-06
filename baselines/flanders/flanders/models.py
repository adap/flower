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


class MnistNet(nn.Module):
    """Neural network for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """Forward pass through the network."""
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train_mnist(model, dataloader, epochs, device):
    """Train the network on the training set."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Step [{i+1}/{len(dataloader)}], "
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


class FMnistNet(nn.Module):
    """Neural network for Fashion MNIST classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        # Dropout module with a 0.2 drop probability
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        """Forward pass through the network."""
        # Flatten the input tensor
        x = x.view(x.shape[0], -1)
        # Set the activation functions
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


def train_fmnist(model, dataloader, epochs, device):
    """Train the network on the training set."""
    criterion = nn.NLLLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    for epoch in range(epochs):
        for i, (images, labels) in enumerate(dataloader):
            images = images.view(-1, 28 * 28).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], "
                    f"Step [{i+1}/{len(dataloader)}], "
                    f"Loss: {loss.item():.4f}"
                )


# pylint: disable=too-many-locals
def test_fmnist(model, dataloader, device):
    """Validate the network on the entire test set."""
    loss = 0
    model.eval()
    criterion = nn.NLLLoss(reduction="sum")
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
