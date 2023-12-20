"""Dataset loading and processing utilities."""

import pickle
from typing import List, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fedpara.dataset_preparation import iid, noniid, DatasetSplit


def load_datasets(
        config, num_clients, batch_size
) -> Tuple[List[DataLoader], DataLoader]:
    """Load the dataset and return the dataloaders for the clients and the server."""
    print("Loading data...")
    if config.name == "CIFAR10":
        Dataset = datasets.CIFAR10
    elif config.name == "CIFAR100":
        Dataset = datasets.CIFAR100
    else:
        raise NotImplementedError
    data_directory = f"./data/{config.name.lower()}/"
    ds_path = f"{data_directory}train_{num_clients}_{config.alpha:.2f}.pkl"
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
    try:
        with open(ds_path, "rb") as file:
            train_datasets = pickle.load(file)
    except FileNotFoundError:
        dataset_train = Dataset(
            data_directory, train=True, download=True, transform=transform_train
        )
        if config.partition == "iid":
            train_datasets = iid(dataset_train, num_clients)
        else:
            train_datasets, _ = noniid(dataset_train, num_clients, config.alpha)
    dataset_test = Dataset(
        data_directory, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2)
    train_loaders = [
        DataLoader(
            DatasetSplit(dataset_train, ids),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        for ids in train_datasets.values()
    ]

    return train_loaders, test_loader
