"""Dataset loading and processing utilities."""

import pickle
from typing import List, Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from fedpara.dataset_preparation import (
    DatasetSplit,
    iid,
    noniid,
    noniid_partition_loader,
)


def load_datasets(
    config, num_clients, batch_size
) -> Tuple[List[DataLoader], DataLoader]:
    """Load the dataset and return the dataloaders for the clients and the server."""
    print("Loading data...")
    match config["name"]:
        case "CIFAR10":
            Dataset = datasets.CIFAR10
        case "CIFAR100":
            Dataset = datasets.CIFAR100
        case "MNIST":
            Dataset = datasets.MNIST
        case _:
            raise NotImplementedError
    data_directory = f"./data/{config['name'].lower()}/"
    match config["name"]:
        case "CIFAR10" | "CIFAR100":
            ds_path = f"{data_directory}train_{num_clients}_{config.alpha:.2f}.pkl"
            transform_train = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            transform_test = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
            try:
                with open(ds_path, "rb") as file:
                    train_datasets = pickle.load(file).values()
                dataset_train = Dataset(
                    data_directory,
                    train=True,
                    download=False,
                    transform=transform_train,
                )
                dataset_test = Dataset(
                    data_directory,
                    train=False,
                    download=False,
                    transform=transform_test,
                )
            except FileNotFoundError:
                dataset_train = Dataset(
                    data_directory, train=True, download=True, transform=transform_train
                )
                if config.partition == "iid":
                    train_datasets = iid(dataset_train, num_clients)
                else:
                    train_datasets, _ = noniid(dataset_train, num_clients, config.alpha)
                pickle.dump(train_datasets, open(ds_path, "wb"))
                train_datasets = train_datasets.values()
                dataset_test = Dataset(
                    data_directory, train=False, download=True, transform=transform_test
                )

        case "MNIST":
            ds_path = f"{data_directory}train_{num_clients}.pkl"
            transform_train = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )
            transform_test = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
            try:
                train_datasets = pickle.load(open(ds_path, "rb"))
                dataset_train = Dataset(
                    data_directory,
                    train=True,
                    download=False,
                    transform=transform_train,
                )
                dataset_test = Dataset(
                    data_directory,
                    train=False,
                    download=False,
                    transform=transform_test,
                )

            except FileNotFoundError:
                dataset_train = Dataset(
                    data_directory, train=True, download=True, transform=transform_train
                )
                train_datasets = noniid_partition_loader(
                    dataset_train,
                    m_per_shard=config.shard_size,
                    n_shards_per_client=len(dataset_train) // (config.shard_size * 100),
                )
                pickle.dump(train_datasets, open(ds_path, "wb"))
                dataset_test = Dataset(
                    data_directory, train=False, download=True, transform=transform_test
                )
            train_loaders = [
                DataLoader(x, batch_size=batch_size, shuffle=True)
                for x in train_datasets
            ]
            test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2)
            return train_loaders, test_loader

    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=2)
    train_loaders = [
        DataLoader(
            DatasetSplit(dataset_train, ids),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        for ids in train_datasets
    ]

    return train_loaders, test_loader
