"""MNIST dataset utilities for federated learning."""


from typing import List, Optional, Tuple

from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, random_split

from flwr_baselines.publications.fedprox_mnist.dataset_preparation import _partition_data

def load_datasets(  # pylint: disable=too-many-arguments
    num_clients: int,
    config: DictConfig,
    val_ratio: float = 0.1,
    batch_size: Optional[int] = 32,
    seed: Optional[int] = 42,
) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    """Creates the dataloaders to be fed into the model.

    Parameters
    ----------
    num_clients : int, optional
        The number of clients that hold a part of the data, by default 10
    config: DictConfig
        A config that parameterises how the dataset should be partitioned.
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
    print(f"Dataset partitioned according to: {config = } and using {num_clients = }")
    datasets, testset = _partition_data(num_clients, config.iid, config.power_law, config.balance, seed)
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for dataset in datasets:
        len_val = int(len(dataset) / (1 / val_ratio))
        lengths = [len(dataset) - len_val, len_val]
        ds_train, ds_val = random_split(
            dataset, lengths, torch.Generator().manual_seed(seed)
        )
        trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    return trainloaders, valloaders, DataLoader(testset, batch_size=batch_size)
