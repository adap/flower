"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""


from typing import Optional, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from dataset_preparation import _partition_data

def load_datasets(  # pylint: disable=too-many-arguments
    config: DictConfig,
    num_clients: int,
    seed: Optional[int] = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Creates the dataloaders to be fed into the model.

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
    Tuple[DataLoader, DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for validation, the DataLoader for testing.
    """
    print(f"Dataset partitioning config: {config}")
    datasets, label_split , client_testsets, testset = _partition_data(
        num_clients,
        dataset_name = config.dataset_name,
        iid = config.iid,
        balance = config.balance,
        seed = seed,
    )
    # Split each partition into train/val and create DataLoader
    trainloaders = []
    for dataset in datasets:
        trainloaders.append(DataLoader(dataset, batch_size=config.batch_size.train, shuffle=True))
    return trainloaders, label_split, client_testsets , DataLoader(testset, batch_size=config.batch_size.test)