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
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr.common import Parameters
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner
from flwr_datasets.preprocessor import Merger

# Type aliases
DataLoaders = Tuple[DataLoader, DataLoader]

# Constants
NUM_CLASSES = 100
CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD = (0.5, 0.5, 0.5)

'''
MobileNet in PyTorch.
See the paper "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"
for more details.
'''


class Block(nn.Module):
    """Implements a MobileNet block with depthwise and pointwise convolutions.

    This block is the fundamental building block of MobileNet architecture,
    implementing the depthwise separable convolution that significantly reduces
    computational cost compared to standard convolutions.

    The block consists of two main operations:
    1. Depthwise convolution: applies a single filter per input channel
    2. Pointwise convolution: 1x1 convolution to change the number of channels

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride for depthwise convolution. Defaults to 1

    Note:
        - BatchNorm is used after each convolution with track_running_stats=False
          to support federated learning scenarios
        - ReLU activation is applied after each BatchNorm
        - All convolutions are bias-free as BatchNorm is used
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_planes,  # Depthwise convolution
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_planes, track_running_stats=False)
        self.conv2 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=1,  # Pointwise convolution
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_planes, track_running_stats=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetCifar(nn.Module):
    """MobileNet architecture for CIFAR-10 dataset.

    The network consists of depthwise separable convolutions which significantly
    reduce the number of parameters compared to standard convolutions while
    maintaining good performance.

    Args:
        num_classes (int, optional): Number of output classes. Defaults to 10.
    """

    # Architecture configuration
    # Format: (channels, stride). If single int, stride=1
    cfg = [
        64,  # Conv 64, stride 1
        (128, 2),  # Conv 128, stride 2
        128,  # Conv 128, stride 1
        (256, 2),  # Conv 256, stride 2
        256,  # Conv 256, stride 1
        (512, 2),  # Conv 512, stride 2
        512,
        512,
        512,
        512,
        512,  # 5x Conv 512, stride 1
        (1024, 2),  # Conv 1024, stride 2
        1024,  # Conv 1024, stride 1
    ]

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.ReLU(inplace=True),
            *self._make_layers(in_planes=32),
            nn.AvgPool2d(2),
            nn.Flatten(),
        )

        # Classification layer
        self.classifier = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes: int) -> List[nn.Module]:
        """Constructs the convolutional layers based on the configuration.

        Args:
            in_planes (int): Number of input channels for the first layer

        Returns:
            List[nn.Module]: List of Block modules forming the network backbone
        """
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        return self.classifier(self.feature_extractor(x))

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the input using only the feature extractor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, H, W)

        Returns:
            torch.Tensor: Features of shape (batch_size, 1024)
        """
        return self.feature_extractor(x)


fds = None  # Cache FederatedDataset


def load_data(
    partition_id: int,
    num_partitions: int,
    num_classes_per_client: int,
    train_test_split_ratio: float,
    batch_size: int,
    seed: int,
) -> DataLoaders:
    """Load and prepare CIFAR-100 data for federated learning.

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
            partition_by="fine_label",
            num_classes_per_partition=num_classes_per_client,
            seed=seed,
        )
        fds = FederatedDataset(
            dataset="uoft-cs/cifar100",
            partitioners={"train": partitioner},
            preprocessor=Merger({"train": ("train", "test")}),
        )

    # Load partition and split into train/test
    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(
        test_size=train_test_split_ratio, seed=42
    )

    # Define image transformations
    pytorch_transforms = Compose(
        [ToTensor(), Normalize(mean=CIFAR_MEAN, std=CIFAR_STD)]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(
        partition_train_test["train"], batch_size=batch_size, shuffle=True
    )
    testloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
    return trainloader, testloader


def train(
    net: MobileNetCifar,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
) -> float:
    """Train the model using FedBABU strategy.

    This method implements the core FedBABU training logic:
    1. Freezes the classifier (head) by setting its learning rate to 0
    2. Trains only the feature extractor (body) with the specified learning rate
    3. Uses SGD optimizer with momentum for training

    The training follows the FedBABU strategy where only the feature extractor
    is trained while keeping the classifier frozen, helping learn generalizable
    features across clients.

    Args:
        net (MobileNetCifar): The neural network model with separate body and head
        trainloader (DataLoader): DataLoader for the training dataset
        epochs (int): Number of training epochs
        lr (float): Learning rate for the feature extractor
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
                "lr": 0.0,  # Freeze classifier weights
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
            labels = batch["fine_label"].to(device)

            # Forward and backward passes
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss / num_batches


def test(
    net: MobileNetCifar,
    testloader: DataLoader,
    trainloader: DataLoader,
    device: torch.device,
    finetune_epochs: int,
    lr: float,
    momentum: float,
) -> Tuple[float, float]:
    """Evaluate model performance with local fine-tuning.

    The evaluation process in FedBABU consists of two steps:
    1. Fine-tune the entire model (both body and head) on local training data
    2. Evaluate the fine-tuned model on local test data

    This approach allows each client to adapt the shared feature extractor to
    their local data distribution while maintaining the benefits of federated
    learning.

    Args:
        net (MobileNetCifar): The neural network model to evaluate
        testloader (DataLoader): DataLoader for test/validation data
        trainloader (DataLoader): DataLoader for fine-tuning
        device (torch.device): Device to run computations on (CPU/GPU)
        finetune_epochs (int): Number of epochs for fine-tuning
        lr (float): Learning rate for fine-tuning
        momentum (float): Momentum factor for SGD optimizer

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
            labels = batch["fine_label"].to(device)

            outputs = net(images)
            total_loss += criterion(outputs, labels).item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()

    avg_loss = total_loss / len(testloader)
    accuracy = correct / total
    return avg_loss, accuracy


def finetune(
    net: MobileNetCifar,
    trainloader: DataLoader,
    finetune_epochs: int,
    lr: float,
    momentum: float,
    device: torch.device,
) -> None:
    """Fine-tune the entire model on local data.

    This method implements a key component of FedBABU where both feature extractor
    and classifier are fine-tuned together. The process involves:
    1. Setting up SGD optimizer for all model parameters
    2. Training the entire model on local data
    3. Allowing the model to adapt to local data distribution

    Unlike the training phase where only the feature extractor is updated,
    fine-tuning updates all model parameters to better fit local data patterns.

    Args:
        net (MobileNetCifar): The neural network model to fine-tune
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
            labels = batch["fine_label"].to(device)

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
