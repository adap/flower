"""Task utilities for MNIST model, data loading, training, and evaluation."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms


class MNISTClassifier(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self, output_size: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 1000)
        self.fc2 = nn.Linear(1000, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def _mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def _load_train_dataset():
    return datasets.MNIST(
        root="./dataset/mnist",
        train=True,
        download=True,
        transform=_mnist_transform(),
    )


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partitioned MNIST data and split into train/validation loaders."""

    full_train = _load_train_dataset()

    # Deterministic partitioning by index shuffling.
    indices = np.arange(len(full_train))
    rng = np.random.default_rng(seed=42)
    rng.shuffle(indices)
    partition_indices = np.array_split(indices, num_partitions)[partition_id].tolist()

    partition_dataset = Subset(full_train, partition_indices)
    val_size = max(1, int(0.2 * len(partition_dataset)))
    train_size = len(partition_dataset) - val_size
    train_subset, val_subset = random_split(
        partition_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_subset, batch_size=batch_size)
    return trainloader, valloader


def load_centralized_test(batch_size: int = 256):
    """Load centralized MNIST test data."""

    testset = datasets.MNIST(
        root="./dataset/mnist",
        train=False,
        download=True,
        transform=_mnist_transform(),
    )
    return DataLoader(testset, batch_size=batch_size)


def train(model, trainloader, epochs: int, lr: float, device: torch.device):
    """Train the model and return average loss."""

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    total_loss = 0.0
    num_batches = 0

    for _ in range(epochs):
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(1, num_batches)


def evaluate(model, dataloader, device: torch.device):
    """Evaluate the model and return average loss and accuracy."""

    criterion = nn.CrossEntropyLoss().to(device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = correct / max(1, total)
    return avg_loss, accuracy
