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

from typing import List, Optional, Tuple

import os
import wget
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from zipfile import ZipFile
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.datasets import CIFAR10


def _download_data(dataset: str) -> Tuple[Dataset, Dataset]:
    """Downloads (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[Dataset, Dataset]
        The training and testing datasets.
    """
    if dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = CIFAR10(
            root="./dataset/cifar10", train=True, download=True, transform=transform
        )
        testset = CIFAR10(
            root="./dataset/cifar10", train=False, download=True, transform=transform
        )
        return trainset, testset
    elif dataset == "FLICKR-AES":
        # Get zip file from 
        # https://drive.google.com/file/d/1jY7GMMNaQGQ80AAL99FLrBpWPFCwTKVT/view?usp=drive_link
        if not os.path.exists("./dataset/FLICKR-AES"):
            os.makedirs("./dataset/FLICKR-AES")
        if not os.path.exists("./dataset/FLICKR-AES/FLICKR-AES.zip"):
            wget.download(
                "https://drive.google.com/u/0/uc?id=1jY7GMMNaQGQ80AAL99FLrBpWPFCwTKVT&export=download",
                "./dataset/FLICKR-AES/FLICKR-AES.zip",
            )
        with ZipFile("./dataset/FLICKR-AES/FLICKR-AES.zip", "r") as zipObj:
            zipObj.extractall("./dataset/FLICKR-AES")

        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        trainset = ImageFolder(
            "./dataset/FLICKR-AES/FLICKR-AES/train", transform=transform
        )
        testset = ImageFolder(
            "./dataset/FLICKR-AES/FLICKR-AES/test", transform=transform
        )
        return trainset, testset

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented.")


def _partition_data(
    num_clients,
    datasets: List[str],
    iid: Optional[bool] = False,
    power_law: Optional[bool] = True,
    balance: Optional[bool] = False,
    seed: Optional[int] = 42,
) -> Tuple[List[Dataset], Dataset]:
    """Split training set into iid or non iid partitions to simulate the
    federated setting.

    Parameters
    ----------
    num_clients : int
        The number of clients that hold a part of the data
    datasets : List[str]
        The datasets to be used for training
    iid : bool, optional
        Whether the data should be independent and identically distributed between
        the clients or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario), by default False
    power_law: bool, optional
        Whether to follow a power-law distribution when assigning number of samples
        for each client, defaults to True
    balance : bool, optional
        Whether the dataset should contain an equal number of samples in each class,
        by default False
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42

    Returns
    -------
    Tuple[List[Dataset], Dataset]
        A list of dataset for each client and a single dataset to be use for testing the model.
    """
    trainsets, testsets = [], []
    for dataset in datasets:
        trainset, testset = _download_data(dataset=dataset)
        trainsets.append(trainset)
        testsets.append(testset)
    print("trainsets", trainsets)
    print("testsets", testsets)

    if balance:
        trainset = _balance_classes(trainset, seed)
    
    partition_size = int(len(trainset) / num_clients)
    lengths = [partition_size] * num_clients
    if sum(lengths) < len(trainset):
        remaining_data = len(trainset) - sum(lengths)
        lengths[-1] += remaining_data

    if iid:
        datasets = random_split(trainset, lengths, torch.Generator().manual_seed(seed))
    else:
        if power_law:
            trainset_sorted = _sort_by_class(trainset)
            datasets = _power_law_split(
                trainset_sorted,
                num_partitions=num_clients,
                num_labels_per_partition=2,
                min_data_per_partition=10,
                mean=0.0,
                sigma=2.0,
            )
        else:
            shard_size = int(partition_size / 2)
            idxs = trainset.targets.argsort()
            sorted_data = Subset(trainset, idxs)
            tmp = []
            for idx in range(num_clients * 2):
                tmp.append(
                    Subset(
                        sorted_data, np.arange(shard_size * idx, shard_size * (idx + 1))
                    )
                )
            idxs_list = torch.randperm(
                num_clients * 2, generator=torch.Generator().manual_seed(seed)
            )
            datasets = [
                ConcatDataset((tmp[idxs_list[2 * i]], tmp[idxs_list[2 * i + 1]]))
                for i in range(num_clients)
            ]

    return datasets, testset


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    """Balance the classes of the trainset.

    Trims the dataset so each class contains as many elements as the
    class that contained the least elements.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be balanced.
    seed : int, optional
        Used to set a fix seed to replicate experiments, by default 42.

    Returns
    -------
    Dataset
        The balanced training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    smallest = np.min(class_counts)
    idxs = trainset.targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [trainset.targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(trainset.targets[idxs[int(count) : int(count + smallest)]])
    unshuffled = ConcatDataset(tmp)
    unshuffled_targets = torch.cat(tmp_targets)
    shuffled_idxs = torch.randperm(
        len(unshuffled), generator=torch.Generator().manual_seed(seed)
    )
    shuffled = Subset(unshuffled, shuffled_idxs)
    shuffled.targets = unshuffled_targets[shuffled_idxs]

    return shuffled


def _sort_by_class(
    trainset: Dataset,
) -> Dataset:
    """Sort dataset by class/label.

    Parameters
    ----------
    trainset : Dataset
        The training dataset that needs to be sorted.

    Returns
    -------
    Dataset
        The sorted training dataset.
    """
    class_counts = np.bincount(trainset.targets)
    idxs = trainset.targets.argsort()  # sort targets in ascending order

    tmp = []  # create subset of smallest class
    tmp_targets = []  # same for targets

    start = 0
    for count in np.cumsum(class_counts):
        tmp.append(
            Subset(trainset, idxs[start : int(count + start)])
        )  # add rest of classes
        tmp_targets.append(trainset.targets[idxs[start : int(count + start)]])
        start += count
    sorted_dataset = ConcatDataset(tmp)  # concat dataset
    sorted_dataset.targets = torch.cat(tmp_targets)  # concat targets
    return sorted_dataset


def _power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> Dataset:
    """Partitions the dataset following a power-law distribution. It follows
    the implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with
    default values set accordingly.

    Parameters
    ----------
    sorted_trainset : Dataset
        The training dataset sorted by label/class.
    num_partitions: int
        Number of partitions to create
    num_labels_per_partition: int
        Number of labels to have in each dataset partition. For
        example if set to two, this means all training examples in
        a given partition will be long to the same two classes. default 2
    min_data_per_partition: int
        Minimum number of datapoints included in each partition, default 10
    mean: float
        Mean value for LogNormal distribution to construct power-law, default 0.0
    sigma: float
        Sigma value for LogNormal distribution to construct power-law, default 2.0

    Returns
    -------
    Dataset
        The partitioned training dataset.
    """

    targets = sorted_trainset.targets
    full_idx = range(len(targets))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx = []
    num_classes = len(np.bincount(targets))
    hist = np.zeros(num_classes, dtype=np.int32)

    # assign min_data_per_partition
    min_data_per_class = int(min_data_per_partition / num_labels_per_partition)
    for u_id in range(num_partitions):
        partitions_idx.append([])
        for cls_idx in range(num_labels_per_partition):
            # label for the u_id-th client
            cls = (u_id + cls_idx) % num_classes
            # record minimum data
            indices = list(
                full_idx[
                    labels_cs[cls]
                    + hist[cls] : labels_cs[cls]
                    + hist[cls]
                    + min_data_per_class
                ]
            )
            partitions_idx[-1].extend(indices)
            hist[cls] += min_data_per_class

    # add remaining images following power-law
    probs = np.random.lognormal(
        mean,
        sigma,
        (num_classes, int(num_partitions / num_classes), num_labels_per_partition),
    )
    remaining_per_class = class_counts - hist
    # obtain how many samples each partition should be assigned for each of the labels it contains
    probs = (
        remaining_per_class.reshape(-1, 1, 1)
        * probs
        / np.sum(probs, (1, 2), keepdims=True)
    )

    for u_id in range(num_partitions):
        for cls_idx in range(num_labels_per_partition):
            cls = (u_id + cls_idx) % num_classes
            count = int(probs[cls, u_id // num_classes, cls_idx])

            # add count of specific class to partition
            indices = full_idx[
                labels_cs[cls] + hist[cls] : labels_cs[cls] + hist[cls] + count
            ]
            partitions_idx[u_id].extend(indices)
            hist[cls] += count

    # construct subsets
    partitions = [Subset(sorted_trainset, p) for p in partitions_idx]
    return partitions
