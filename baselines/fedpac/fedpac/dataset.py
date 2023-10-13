from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from fedpac.dataset_preparation import partition_cifar_data, partition_emnist_data, _download_data
from collections import defaultdict



def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates the dataloaders to be fed into the model.

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
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    print(f"Dataset partitioning config: {config}")
    dataset = config.name
    if dataset=='cifar10':
        trainset, testset = _download_data(dataset)
        datasets = partition_cifar_data(
            trainset,
            num_clients,
            iid=config.iid,
            balance=config.balance,
            s = config.s,
            seed=seed,
        )
        val_datasets = partition_cifar_data(
            testset,
            num_clients,
            iid=config.iid,
            balance=config.balance,
            s = config.s,
            sample_size=300,
            seed=seed,
        )

    elif dataset=='emnist':
        datasets, testset = partition_emnist_data(
            dataset,
            num_clients,
            iid=config.iid,
            balance=config.balance,
            s = config.s,
            seed=seed,
        )       

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []

    for dataset, val_dataset in zip(datasets, val_datasets):
        # len_val = int(len(dataset) / (1 / val_ratio))
        # lengths = [len(dataset) - len_val, len_val]
        # ds_train, ds_val = random_split(
        #     dataset, lengths, torch.Generator().manual_seed(seed)
        # )
        trainloaders.append(DataLoader(dataset, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(val_dataset, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)
