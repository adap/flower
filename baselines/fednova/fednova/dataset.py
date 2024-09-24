"""Dataloaders for the CIFAR-10 dataset."""

from typing import List, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fednova.dataset_preparation import DataPartitioner


def load_datasets(config: DictConfig) -> Tuple[List[DataLoader], DataLoader, List]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    val_ratio : float, optional
        The ratio of training data that will be used for validation (between 0 and 1),
        by default 0.1
    batch_size : int, optional
        The size of the batches to be fed into the model, by default 32
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, List]
        The DataLoader for training, the DataLoader for testing, client dataset sizes.
    """
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = datasets.CIFAR10(
        root=config.datapath, train=True, download=True, transform=transform_train
    )

    testset = datasets.CIFAR10(
        root=config.datapath, train=False, download=True, transform=transform_test
    )

    partition_sizes = [1.0 / config.num_clients for _ in range(config.num_clients)]

    partition_obj = DataPartitioner(
        trainset, partition_sizes, is_non_iid=config.NIID, alpha=config.alpha
    )
    ratio = partition_obj.ratio

    trainloaders = []
    for data_split in range(config.num_clients):
        client_partition = partition_obj.use(data_split)
        trainloaders.append(
            torch.utils.data.DataLoader(
                client_partition,
                batch_size=config.batch_size,
                shuffle=True,
                pin_memory=True,
            )
        )

    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    return trainloaders, test_loader, ratio
