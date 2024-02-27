"""Load the MNIST dataset into train and test loaders."""

from typing import List, Optional, Tuple

from torch.utils.data import DataLoader

from tamuna.dataset_preparation import partition_data


def load_datasets(
    num_clients: int, iid: bool = False, seed: Optional[int] = 42
) -> Tuple[List[DataLoader], DataLoader]:
    """Create dataloaders.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool
        Whether the data should be split in independent identically distributed (iid)
        fashion or not. If False, data will be split according to power law.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[DataLoader], DataLoader]
        The DataLoaders for training, the DataLoader for testing.
    """
    datasets, testset = partition_data(num_clients, iid, seed=seed)

    trainloaders = []
    for dataset in datasets:
        trainloaders.append(DataLoader(dataset, batch_size=len(dataset), shuffle=True))
    return trainloaders, DataLoader(testset, batch_size=len(testset))
