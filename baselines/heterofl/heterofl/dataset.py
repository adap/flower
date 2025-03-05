"""Utilities for creation of DataLoaders for clients and server."""

from typing import List, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from heterofl.dataset_preparation import _partition_data


def load_datasets(  # pylint: disable=too-many-arguments
    strategy_name: str,
    config: DictConfig,
    num_clients: int,
    seed: Optional[int] = 42,
) -> Tuple[
    DataLoader, List[DataLoader], List[torch.tensor], List[DataLoader], DataLoader
]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        Parameterises the dataset partitioning process
    num_clients : int
        The number of clients that hold a part of the data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader, DataLoader]
        The entire trainset Dataloader for testing purposes,
        The DataLoader for training, the DataLoader for validation,
        the DataLoader for testing.
    """
    print(f"Dataset partitioning config: {config}")
    trainset, datasets, label_split, client_testsets, testset = _partition_data(
        num_clients,
        dataset_name=config.dataset_name,
        strategy_name=strategy_name,
        iid=config.iid,
        dataset_division={
            "shard_per_user": config.shard_per_user,
            "balance": config.balance,
        },
        seed=seed,
    )
    # Split each partition into train/val and create DataLoader
    entire_trainloader = DataLoader(
        trainset, batch_size=config.batch_size.train, shuffle=config.shuffle.train
    )

    trainloaders = []
    valloaders = []
    for dataset in datasets:
        trainloaders.append(
            DataLoader(
                dataset,
                batch_size=config.batch_size.train,
                shuffle=config.shuffle.train,
            )
        )

    for client_testset in client_testsets:
        valloaders.append(
            DataLoader(
                client_testset,
                batch_size=config.batch_size.test,
                shuffle=config.shuffle.test,
            )
        )

    return (
        entire_trainloader,
        trainloaders,
        label_split,
        valloaders,
        DataLoader(
            testset, batch_size=config.batch_size.test, shuffle=config.shuffle.test
        ),
    )
