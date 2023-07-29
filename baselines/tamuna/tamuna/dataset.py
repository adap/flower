from typing import Optional, Tuple, List

from torch.utils.data import DataLoader
from tamuna.dataset_preparation import _partition_data


def load_datasets(num_clients: int, seed: Optional[int] = 42) -> Tuple[List[DataLoader], DataLoader]:
    """
    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for testing.
    """
    datasets, testset = _partition_data(num_clients, seed=seed)

    trainloaders = []
    for dataset in datasets:
        trainloaders.append(DataLoader(dataset, batch_size=len(dataset), shuffle=True))
    return trainloaders, DataLoader(testset, batch_size=len(testset))
