from typing import Optional, Tuple, List

from torch.utils.data import DataLoader
from dataset_preparation import partition_data


def load_datasets(
        num_clients: int,
        iid: bool = False,
        seed: Optional[int] = 42
) -> Tuple[List[DataLoader], DataLoader]:
    """
    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    iid : bool
        Whether the data should be split in independent identically distributed (iid) fashion or not.
        If False, data will be split according to power law.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        The DataLoader for training, the DataLoader for testing.
    """
    datasets, testset = partition_data(num_clients, iid, seed=seed)

    trainloaders = []
    for dataset in datasets:
        trainloaders.append(DataLoader(dataset, batch_size=len(dataset), shuffle=True))
    return trainloaders, DataLoader(testset, batch_size=len(testset))
