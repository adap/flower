from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import CIFAR100


def _download_data() -> Tuple[Dataset, Dataset]:
    """Downloads (if necessary) and returns the CIFAR-100 dataset.

    Returns
    -------
    Tuple[CIFAR100, CIFAR100]
        The dataset for training and the dataset for testing CIFAR100.
    """
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761))])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                            (0.2675, 0.2565, 0.2761))])

    trainset = CIFAR100("./dataset", train=True, download=True, transform=transform_train)
    testset = CIFAR100("./dataset", train=False, download=True, transform=transform_test)
    return trainset, testset


def _partition_data(
    num_clients,
    iid: Optional[bool] = True,
    seed: Optional[int] = 41,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default False
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """
    trainset, testset = _download_data()

    datasets = list()

    if iid:

        np.random.seed(seed)
        num_sample = int(len(trainset)/(num_clients))
        index = [i for i in range(len(trainset))]
        for i in range(num_clients):
            sample_idx = np.random.choice(index, num_sample,
                                            replace=False)
            index = list(set(index)-set(sample_idx))
            datasets.append(Subset(trainset, sample_idx))

    else:
        pass

    return datasets, testset