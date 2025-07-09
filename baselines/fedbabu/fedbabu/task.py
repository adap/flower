"""FedBABU implementation in PyTorch using Flower.

This module implements the FedBABU (Federated Learning with Body and Head) approach,
as described in the paper "FedBABU: Towards Enhanced Representation Learning in
Federated Learning via Backbone Update" (https://arxiv.org/abs/2302.09597).

Key components:
- MobileNet architecture adapted for CIFAR-10
- Federated learning with body-head separation
- Local fine-tuning for personalization
- Non-IID data handling using Dirichlet distribution

The approach involves:
1. Training the feature extractor (body) in a federated way
2. Keeping the classifier (head) frozen during federated training
3. Fine-tuning both body and head locally for each client
"""

from collections import OrderedDict
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Parameters
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from flwr_datasets.preprocessor import Merger
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)

# Type aliases
DataLoaders = Tuple[DataLoader, DataLoader]

# Constants
NUM_CLASSES = 10
CIFAR_MEAN = (0.485, 0.456, 0.406)
CIFAR_STD = (0.229, 0.224, 0.225)

'''
MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''

import torch
import torch.nn as nn


class FourConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(FourConvNet, self).__init__()

        # Feature extraction part
        self.feature_extractor = nn.Sequential(
            # First convolutional block
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False
                ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
                ),
            ),
            # Second convolutional block
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False
                ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
                ),
            ),
            # Third convolutional block
            nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(
                    64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False
                ),
                nn.ReLU(),
                nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
                ),
            ),
        )

        # Classifier part
        self.classifier = nn.Linear(
            1024, num_classes
        )  # Note: input feature dimension should match actual feature map size

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # Flatten the feature map
        x = self.classifier(x)
        return x


fds = None  # Cache FederatedDataset


def load_data(
    partition_id: int,
    num_partitions: int,
    num_classes_per_client: int,
    train_test_split_ratio: float,
    batch_size: int,
    seed: int,
) -> DataLoaders:
    """Load and prepare CIFAR-10 data for federated learning.

    This function handles dataset preparation through the following steps:
    1. Creates non-IID data partitions using the PathologicalPartitioner
    2. Loads and splits data for the specified client partition
    3. Applies necessary data transformations
    4. Creates DataLoader instances for training and testing

    The partitioning ensures that each client receives a specific subset of
    classes, simulating a realistic federated learning scenario with non-IID
    data distribution.

    Args:
        partition_id (int): ID of the client's partition to load
        num_partitions (int): Total number of partitions across all clients
        num_classes_per_client (int): Number of unique classes per client
        train_test_split_ratio (float): Ratio of data to use for testing
        batch_size (int): Number of samples per batch
        seed (int): Random seed for reproducibility

    Returns:
        DataLoaders: Contains:
            - DataLoader for training data
            - DataLoader for testing data

    Note:
        Uses a global FederatedDataset instance to avoid reloading data
        for each client in the same process.
    """
    global fds
    if fds is None:
        # Initialize dataset with Dirichlet partitioning for non-IID data distribution
        partitioner = PathologicalPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            num_classes_per_partition=num_classes_per_client,
            seed=seed,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
            preprocessor=Merger({"train": ("train", "test")}),
        )

    # Load partition and split into train/test
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(
        test_size=train_test_split_ratio, seed=42
    )

    # Define image transformations
    cifar10_transforms_train = Compose(
        [
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
        ]
    )
    cifar10_transforms_test = Compose(
        [ToTensor(), Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)]
    )

    def apply_transforms_train(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [cifar10_transforms_train(img) for img in batch["img"]]
        return batch

    def apply_transforms_test(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [cifar10_transforms_test(img) for img in batch["img"]]
        return batch

    partition_train_test["train"] = partition_train_test["train"].with_transform(
        apply_transforms_train
    )
    partition_train_test["test"] = partition_train_test["test"].with_transform(
        apply_transforms_test
    )
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def train(
    algorithm: str,
    net: FourConvNet,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
) -> float:
    """Train the model on the local dataset using FedBABU or FedAvg.

    This method implements two training strategies based on the algorithm argument:

    FedBABU:
    1. Freezes the classifier (head) by setting its learning rate to 0
    2. Trains only the feature extractor (body) with the specified learning rate
    3. Uses SGD optimizer with momentum for training

    FedAvg:
    1. Trains the entire model (both body and head) with the specified learning rate
    2. Uses SGD optimizer with momentum for training

    Args:
        algorithm (str): Training strategy to use ("fedbabu" or "fedavg")
        net (FourConvNet): The neural network model with separate body and head
        trainloader (DataLoader): DataLoader for the training dataset
        epochs (int): Number of training epochs
        lr (float): Learning rate for the optimizer
        momentum (float): Momentum factor for SGD optimizer
        device (torch.device): Device to run computations on (CPU/GPU)

    Returns:
        float: Average training loss across all batches and epochs
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # FedBABU: Freeze classifier (head) during training
    optimizer = torch.optim.SGD(
        [
            {"params": net.feature_extractor.parameters(), "lr": lr},
            {
                "params": net.classifier.parameters(),
                "lr": (
                    0.0 if algorithm == "fedbabu" else lr
                ),  # Freeze classifier weights if algorithm is FedBABU
            },
        ],
        momentum=momentum,
    )

    net.train()
    running_loss = 0.0
    num_batches = len(trainloader)

    for _ in range(epochs):
        for batch in trainloader:
            # Move data to appropriate device
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            # Forward and backward passes
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss / num_batches


def test(
    net: FourConvNet,
    testloader: DataLoader,
    trainloader: DataLoader,
    device: torch.device,
    finetune_epochs: int,
    momentum: float,
    lr: float,
) -> Tuple[float, float]:
    """Evaluate the model on local validation data after optional fine-tuning.

    The evaluation process is designed for FedBABU but can be used for FedAvg as well:

    FedBABU:
    1. Fine-tune the entire model (both body and head) on local training data
    2. Evaluate the fine-tuned model on local validation data
    This allows the model to adapt to local data distributions while
    maintaining the benefits of federated feature learning.

    FedAvg:
    1. (Optionally) skip fine-tuning and evaluate the model as-is
    2. Evaluate directly on local validation data

    Args:
        net (FourConvNet): The neural network model to evaluate
        testloader (DataLoader): DataLoader for test/validation data
        trainloader (DataLoader): DataLoader for fine-tuning (FedBABU)
        device (torch.device): Device to run computations on (CPU/GPU)
        finetune_epochs (int): Number of epochs for fine-tuning (FedBABU)
        momentum (float): Momentum factor for SGD optimizer (FedBABU)
        lr (float): Learning rate for fine-tuning

    Returns:
        Tuple[float, float]: Contains:
            - Average loss on test dataset
            - Classification accuracy on test dataset
    """
    # First fine-tune the model on local data
    finetune(net, trainloader, finetune_epochs, lr, momentum, device)

    # Evaluate the fine-tuned model
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = len(testloader.dataset)

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    return avg_loss, accuracy


def finetune(
    net: FourConvNet,
    trainloader: DataLoader,
    finetune_epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
) -> None:
    """Fine-tune the entire model on local data (FedBABU personalization step).

    This method fine-tunes both the feature extractor and classifier on local data.
    Used in FedBABU for local adaptation before evaluation.

    Args:
        net (FourConvNet): The neural network model to fine-tune
        trainloader (DataLoader): DataLoader for the training dataset
        finetune_epochs (int): Number of fine-tuning epochs
        lr (float): Learning rate for fine-tuning
        momentum (float): Momentum factor for SGD optimizer
        device (torch.device): Device to run computations on (CPU/GPU)
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    # Unlike training, we fine-tune all parameters including the classifier
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

    net.train()
    for _ in range(finetune_epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


def get_weights(net: nn.Module) -> Parameters:
    """Extract model weights for federated learning communication.

    This function prepares model parameters for federation by:
    1. Extracting parameters from the model's state dictionary
    2. Converting PyTorch tensors to NumPy arrays
    3. Maintaining consistent parameter ordering

    Args:
        net (nn.Module): PyTorch model to extract weights from

    Returns:
        Parameters: List of NumPy arrays containing model parameters in the
            order they appear in the model's state dictionary

    Note:
        Ensures parameters are moved to CPU before conversion to NumPy arrays
    """
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: Parameters) -> None:
    """Update model weights from federated learning aggregation.

    This function updates the model's parameters by:
    1. Converting NumPy arrays to PyTorch tensors
    2. Creating an ordered state dictionary
    3. Loading parameters into the model with strict checking

    Args:
        net (nn.Module): PyTorch model to update
        parameters (Parameters): List of NumPy arrays containing model parameters
            in the same order as returned by get_weights()

    Note:
        Enforces strict loading to ensure all parameters are properly updated
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
