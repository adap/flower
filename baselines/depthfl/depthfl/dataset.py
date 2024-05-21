"""CIFAR100 dataset utilities for federated learning."""

from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split

from depthfl.dataset_preparation import _partition_data


def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    val_ratio: float = 0.0,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 41,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, validation, and testing.
    """
    print(f"Dataset partitioning config: {config}")
    datasets, testset = _partition_data(
        num_clients,
        iid=config.iid,
        beta=config.beta,
        seed=seed,
    )
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = 0
        if val_ratio > 0:
            len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)
