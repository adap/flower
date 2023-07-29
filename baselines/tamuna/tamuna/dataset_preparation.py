from typing import List, Optional, Tuple

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split
from torchvision.datasets import MNIST


def _download_data() -> Tuple[Dataset, Dataset]:
    """
    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def _partition_data(num_clients, seed: Optional[int] = 42) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid partitions to simulate the federated setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be used for testing the model.
    """
    trainset, testset = _download_data()

    partition_size = int(len(trainset) / num_clients)
    lengths = [partition_size] * num_clients

    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))

    return datasets, testset
