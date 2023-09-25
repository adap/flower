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
import torch
from typing import List, Tuple, Dict
from collections import Counter
from torch.utils.data import Dataset, Subset, ConcatDataset
from torchvision.datasets import EMNIST, CIFAR10, MNIST, FashionMNIST
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

def _download_data(dataset_name = "emnist") -> Tuple[Dataset, Dataset]:
    """Downloads the requested dataset. Currently supports emnist and cifar10

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training dataset, the test dataset.
    """
    trainset, testset = None, None
    if dataset_name == "emnist":
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
    elif dataset_name == "cifar10":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = CIFAR10(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = CIFAR10(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    elif dataset_name == "mnist" or dataset_name == "fashionmnist":
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        trainset = MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform_train,
        )
        testset = MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform_test,
        )
    else:
        raise NotImplementedError

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
    t = trainset.dataset.targets
    if isinstance(t, list):
        t = np.array(t)
    targets = t[trainset.indices]
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
    dataset_name = "emnist"
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
    trainset, testset = _download_data(dataset_name)
    trainsets_per_client = []
    # for s% similarity sample iid data per client
    s_fraction = int(similarity * len(trainset))
    prng = np.random.RandomState(seed)
    idxs = prng.choice(len(trainset), s_fraction, replace=False)
    iid_trainset = Subset(trainset, idxs)
    rem_trainset = Subset(trainset, np.setdiff1d(np.arange(len(trainset)), idxs))

    # sample iid data per client from iid_trainset
    all_ids = np.arange(len(iid_trainset))
    splits = np.array_split(all_ids, num_clients)
    for i in range(num_clients):
        c_ids = splits[i]
        d_ids = iid_trainset.indices[c_ids]
        trainsets_per_client.append(Subset(iid_trainset.dataset, d_ids))

    # sorted_trainset = _sort_by_class(rem_trainset)
    # sample non-iid data per client from rem_trainset by sort method 
    # [logic adapted from this repo](https://github.com/KarhouTam/SCAFFOLD-PyTorch/blob/master/data/utils/partition/assign_classes.py)
    # num_classes = len(rem_trainset.dataset.classes)
    # num_shards = num_clients * num_classes
    # size_shard = len(rem_trainset) // num_shards
    # idx_shard = range(num_shards)
    # for i in range(num_clients):
    #     selected_shard_idx = np.random.choice(idx_shard, num_classes, replace=False)
    #     idx_shard = np.setdiff1d(idx_shard, selected_shard_idx)
    #     t_datasets = [trainsets_per_client[i]]
    #     for shard_idx in selected_shard_idx:
    #         t_ids = np.arange(shard_idx * size_shard, (shard_idx + 1) * size_shard)
    #         d_ids = sorted_trainset.indices[t_ids]
    #         t_datasets.append(Subset(sorted_trainset.dataset, d_ids))
    #     trainsets_per_client[i] = ConcatDataset(t_datasets)

    # sample two classes per client from rem_trainset
    t = rem_trainset.dataset.targets
    if isinstance(t, list):
        t = np.array(t)
    if isinstance(t, torch.Tensor):
        t = t.numpy()
    targets = t[rem_trainset.indices]
    num_remaining_classes = len(set(targets))
    remaining_classes = list(set(targets))
    client_classes = [[] for _ in range(num_clients)]
    times = [0 for _ in range(num_remaining_classes)]

    for i in range(num_clients):
        client_classes[i] = [remaining_classes[i%num_remaining_classes]]
        times[i%num_remaining_classes] += 1
        j = 1
        while j < 2:
            index = prng.choice(num_remaining_classes)
            class_t = remaining_classes[index]
            if class_t not in client_classes[i]:
                client_classes[i].append(class_t)
                times[index] += 1
                j += 1
    
    rem_trainsets_per_client = [[] for _ in range(num_clients)]

    for i in range(num_remaining_classes):
        class_t = remaining_classes[i]
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if class_t in client_classes[j]:
                act_idx = rem_trainset.indices[idx_k_split[ids]]
                rem_trainsets_per_client[j].append(Subset(rem_trainset.dataset, act_idx))
                ids += 1

    for i in range(num_clients):
        trainsets_per_client[i] = ConcatDataset([trainsets_per_client[i]] + rem_trainsets_per_client[i])

    return trainsets_per_client, testset

def _partition_data_dirichlet(
    num_clients,
    alpha,
    seed=42,
    dataset_name = "emnist"
) -> Tuple[List[Dataset], Dataset]:
    """
    Partitions according to the Dirichlet distribution

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    alpha: float
        Parameter of the Dirichlet distribution
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used
    
    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    min_required_samples_per_client = 10
    min_samples = 0
    prng = np.random.RandomState(seed)

    # get the targets
    t = trainset.targets
    if isinstance(t, list):
        t = np.array(t)
    num_classes = len(set(t))
    total_samples = len(t)
    while min_samples < min_required_samples_per_client:
        idx_clients = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.where(t == k)[0]
            prng.shuffle(idx_k)
            proportions = prng.dirichlet(np.repeat(alpha, num_clients))
            proportions = np.array([p * (len(idx_j) < total_samples / num_clients) for p, idx_j in zip(proportions, idx_clients)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_k_split = np.split(idx_k, proportions)
            idx_clients = [idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)]
            min_samples = min([len(idx_j) for idx_j in idx_clients])
        
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset

def _partition_data_label_quantity(
    num_clients,
    labels_per_client,
    seed=42,
    dataset_name = "emnist"
):
    """
    Partitions the data according to the number of labels per client
    logic from https://github.com/Xtra-Computing/NIID-Bench/blob/f10d1a34515cff5a0c1bb85160aa6b10c892bab5/partition.py

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    num_labels_per_client: int
        Number of labels per client
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42
    dataset_name : str
        Name of the dataset to be used
    
    Returns
    -------
    Tuple[List[Subset], Dataset]
        The list of datasets for each client, the test dataset.
    """
    trainset, testset = _download_data(dataset_name)
    prng = np.random.RandomState(seed)

    targets = trainset.targets
    if isinstance(targets, list):
        targets = np.array(targets)
    num_classes = len(set(targets))
    times = [0 for _ in range(num_classes)]
    contains = []

    for i in range(num_clients):
        current = [i%num_classes]
        times[i%num_classes] += 1
        j = 1
        while j < labels_per_client:
            index = prng.randint(0, num_classes)
            if index not in current:
                current.append(index)
                times[index] += 1
                j += 1
        contains.append(current)
    idx_clients = [[] for _ in range(num_clients)]
    for i in range(num_classes):
        idx_k = np.where(targets == i)[0]
        prng.shuffle(idx_k)
        idx_k_split = np.array_split(idx_k, times[i])
        ids = 0
        for j in range(num_clients):
            if i in contains[j]:
                idx_clients[j] += idx_k_split[ids].tolist()
                ids += 1
    trainsets_per_client = [Subset(trainset, idxs) for idxs in idx_clients]
    return trainsets_per_client, testset



if __name__ == "__main__":
    _partition_data(100, 0.1)