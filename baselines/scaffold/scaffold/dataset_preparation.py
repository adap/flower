"""Handle the dataset partitioning and (optionally) complex downloads.

Please add here all the necessary logic to either download, uncompress, pre/post-process
your dataset (or all of the above). If the desired way of running your baseline is to
first download the dataset and partition it and then run the experiments, please
uncomment the lines below and tell us in the README.md (see the "Running the Experiment"
block) that this file should be executed first.
"""
# import hydra
# from hydra.core.hydra_config import HydraConfig
# from hydra.utils import call, instantiate
# from omegaconf import DictConfig, OmegaConf


# @hydra.main(config_path="conf", config_name="base", version_base=None)
# def download_and_preprocess(cfg: DictConfig) -> None:
#     """Does everything needed to get the dataset.

#     Parameters
#     ----------
#     cfg : DictConfig
#         An omegaconf object that stores the hydra config.
#     """

#     ## 1. print parsed config
#     print(OmegaConf.to_yaml(cfg))

#     # Please include here all the logic
#     # Please use the Hydra config style as much as possible specially
#     # for parts that can be customised (e.g. how data is partitioned)

# if __name__ == "__main__":

#     download_and_preprocess()

from typing import List, Tuple
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import EMNIST
import numpy as np
import torchvision.transforms as transforms

def _download_data() -> Tuple[Dataset, Dataset]:
    """Downloads the EMNIST dataset.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training dataset, the test dataset.
    """
    # unsqueeze, flatten
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    trainset = EMNIST(
        root="data",
        split="balanced",
        train=True,
        download=True,
        transform=transform,
    )
    testset = EMNIST(
        root="data",
        split="balanced",
        train=False,
        download=True,
        transform=transform,
    )

    return trainset, testset

def _sort_by_class(
    trainset: Subset,
) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Subset
        The training dataset that needs to be sorted.

    Returns
    -------
    Subset
        The sorted training dataset.
    """

    # get the targets
    targets = trainset.dataset.targets[trainset.indices]
    # get the trainset.indices in the sorted order of the targets
    sorted_idxs = np.argsort(targets)
    # sort the trainset.indices
    t_idx = trainset.indices[sorted_idxs]
    # create a new Subset with the sorted indices
    sorted_dataset = Subset(trainset.dataset, t_idx)
    return sorted_dataset


def _partition_data(
    num_clients,
    similarity=1.0,
    seed=42,
) -> Tuple[List[Dataset], Dataset]:
    """Partitions the dataset into subsets for each client.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    similarity: float
        Parameter to sample similar data
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data()
    trainsets_per_client = []
    # for s% similarity sample iid data per client
    s_fraction = int(similarity * len(trainset))
    np.random.seed(seed)
    idxs = np.random.choice(len(trainset), s_fraction, replace=False)
    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(len(trainset)), idxs))
    # sample iid data per client from iid_trainset
    iid_samples_per_client = [len(iid_trainset) // num_clients] * num_clients
    for i in range(len(iid_samples_per_client)):
        if sum(iid_samples_per_client) < len(iid_trainset):
            iid_samples_per_client[i] += 1
    for i in range(num_clients):
        c_ids = np.random.choice(
            len(iid_trainset), iid_samples_per_client[i], replace=False
        )
        trainsets_per_client.append(Subset(iid_trainset, c_ids))
    # sample non-iid data per client from rem_trainset
    noniid_samples_per_client = [len(rem_trainset) // num_clients] * num_clients
    for i in range(len(noniid_samples_per_client)):
        if sum(noniid_samples_per_client) < len(rem_trainset):
            noniid_samples_per_client[i] += 1

    sorted_trainset = _sort_by_class(rem_trainset)
    start = 0
    for i in range(num_clients):
        end = start + noniid_samples_per_client[i]
        t_ids = np.arange(start, end)
        d_ids = sorted_trainset.indices[t_ids]
        trainsets_per_client[i] = ConcatDataset([trainsets_per_client[i], Subset(sorted_trainset.dataset, d_ids)])
        start = end
    return trainsets_per_client, testset

if __name__ == "__main__":
    _partition_data(100, 0.1)