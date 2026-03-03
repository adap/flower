import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
from typing import List, OrderedDict, Tuple


import datasets
import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
import tqdm
import yaml
from datasets import Dataset
from flwr.common.typing import NDArrays
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (DirichletPartitioner, LinearPartitioner,
                                       SizePartitioner, SquarePartitioner)
from torch.utils.data import DataLoader


# Loading the model (Called when initializing FlowerClient and when testing)
def get_model() -> torch.nn.Module:
    return timm.create_model("resnet18").cuda()


transform_train = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomRotation(15),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ]
)

transform_test = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (0.5071, 0.4867, 0.4408),
            (0.2675, 0.2565, 0.2761),
        ),
    ]
)


def size_based_split(num_clients):
    partitioner = SizePartitioner(
        partition_sizes=[int(50000 / num_clients) for _ in range(num_clients)],
    )
    return partitioner


def dirichlet_based_split(num_clients, alpha):
    partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        min_partition_size=30,
    )
    return FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )


fds = dirichlet_based_split(num_clients=36, alpha=10)


def load_data(
    partition_id: int, num_clients: int = 36, num_workers: int = 4, batch_size: int = 1024
) -> DataLoader:
    partitioner = size_based_split(num_clients)
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
        trust_remote_code=True,
    )

    dataset = fds.load_partition(partition_id=partition_id)

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [transform_train(img) for img in batch["img"]]
        return batch

    dataset = dataset.with_transform(apply_transforms)

    trainloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    return trainloader


# Load the (global) test dataset
def load_global_test_data(batch_size: int = 64) -> DataLoader:
    testset = fds.load_split("test")

    def apply_test_transform(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [transform_test(img) for img in batch["img"]]
        return batch

    testset = testset.with_transform(apply_test_transform)

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    return testloader


# Train and test on a trainloader and testloader
def train(
    model: nn.Module,
    trainloader: DataLoader,
    epochs: int = 5,
    device: str = "cuda",
    lr: float = 1e-3,
    **kwargs,
):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    print("Starting training")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    for _ in tqdm.trange(epochs):
        for batch in trainloader:
            images, labels = batch["img"].to(device), batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def test(
    model: nn.Module,
    testloader: DataLoader,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Validate the model on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    if len(testloader) == 0:
        return np.inf, 0
    with torch.no_grad():
        for batch in testloader: 
            images, labels = batch["img"].to(device), batch["label"].to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    del (testloader, model)
    torch.cuda.empty_cache()
    return loss, accuracy


def ndarrays_from_model(model: torch.nn.ModuleList) -> NDArrays:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def ndarrays_to_model(model: torch.nn.ModuleList, params: NDArrays):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def evaluate_fn(server_round, weights_aggregated, config, **kwargs):
    model = get_model()
    ndarrays_to_model(model, weights_aggregated)
    loss, accuracy = test(model, load_global_test_data())
    del model
    torch.cuda.empty_cache()
    return loss, accuracy


def get_initial_state_dict():
    init_model = get_model()
    initial_state_dict = init_model.state_dict()
    del init_model
    torch.cuda.empty_cache()
    return initial_state_dict
