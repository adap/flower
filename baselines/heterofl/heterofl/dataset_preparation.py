"""Functions for dataset download and processing."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision import transforms

import heterofl.datasets as dt


def _download_data(dataset_name: str, strategy_name: str) -> Tuple[Dataset, Dataset]:
    root = "./data/{}".format(dataset_name)
    if dataset_name == "MNIST":
        trainset = dt.MNIST(
            root=root,
            split="train",
            subset="label",
            transform=dt.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
        testset = dt.MNIST(
            root=root,
            split="test",
            subset="label",
            transform=dt.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        )
    elif dataset_name == "CIFAR10":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        if strategy_name == "heterofl":
            normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        trainset = dt.CIFAR10(
            root=root,
            split="train",
            subset="label",
            transform=dt.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        testset = dt.CIFAR10(
            root=root,
            split="test",
            subset="label",
            transform=dt.Compose(
                [
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
    else:
        raise ValueError(f"{dataset_name} is not valid")

    return trainset, testset


# pylint: disable=too-many-arguments
def _partition_data(
    num_clients: int,
    dataset_name: str,
    strategy_name: str,
    iid: Optional[bool] = False,
    dataset_division=None,
    seed: Optional[int] = 42,
) -> Tuple[Dataset, List[Dataset], List[torch.tensor], List[Dataset], Dataset]:
    trainset, testset = _download_data(dataset_name, strategy_name)

    if dataset_name in ("MNIST", "CIFAR10"):
        classes_size = 10

    if dataset_division["balance"]:
        trainset = _balance_classes(trainset, seed)

    if iid:
        datasets, label_split = iid_partition(trainset, num_clients, seed=seed)
        client_testsets, _ = iid_partition(testset, num_clients, seed=seed)
    else:
        datasets, label_split = non_iid(
            {"dataset": trainset, "classes_size": classes_size},
            num_clients,
            dataset_division["shard_per_user"],
        )
        client_testsets, _ = non_iid(
            {
                "dataset": testset,
                "classes_size": classes_size,
            },
            num_clients,
            dataset_division["shard_per_user"],
            label_split,
        )

        tensor_label_split = []
        for i in label_split:
            tensor_label_split.append(torch.Tensor(i))
        label_split = tensor_label_split

    return trainset, datasets, label_split, client_testsets, testset


def iid_partition(
    dataset: Dataset, num_clients: int, seed: Optional[int] = 42
) -> Tuple[List[Dataset], List[torch.tensor]]:
    """IID partition of dataset among clients."""
    partition_size = int(len(dataset) / num_clients)
    lengths = [partition_size] * num_clients

    divided_dataset = random_split(
        dataset, lengths, torch.Generator().manual_seed(seed)
    )
    label_split = []
    for i in range(num_clients):
        label_split.append(
            torch.unique(torch.Tensor([target for _, target in divided_dataset[i]]))
        )

    return divided_dataset, label_split


def non_iid(
    dataset_info,
    num_clients: int,
    shard_per_user: int,
    label_split=None,
    seed=42,
) -> Tuple[List[Dataset], List]:
    """Non-IID partition of dataset among clients.

    Adopted from authors (of heterofl) implementation.
    """
    data_split: Dict[int, List] = {i: [] for i in range(num_clients)}

    label_idx_split, shard_per_class = _split_dataset_targets_idx(
        dataset_info["dataset"],
        shard_per_user,
        num_clients,
        dataset_info["classes_size"],
    )

    if label_split is None:
        label_split = list(range(dataset_info["classes_size"])) * shard_per_class
        label_split = torch.tensor(label_split)[
            torch.randperm(
                len(label_split), generator=torch.Generator().manual_seed(seed)
            )
        ].tolist()
        label_split = np.array(label_split).reshape((num_clients, -1)).tolist()

        for i, _ in enumerate(label_split):
            label_split[i] = np.unique(label_split[i]).tolist()

    for i in range(num_clients):
        for label_i in label_split[i]:
            idx = torch.arange(len(label_idx_split[label_i]))[
                torch.randperm(
                    len(label_idx_split[label_i]),
                    generator=torch.Generator().manual_seed(seed),
                )[0]
            ].item()
            data_split[i].extend(label_idx_split[label_i].pop(idx))

    return (
        _get_dataset_from_idx(dataset_info["dataset"], data_split, num_clients),
        label_split,
    )


def _split_dataset_targets_idx(dataset, shard_per_user, num_clients, classes_size):
    label = np.array(dataset.target) if hasattr(dataset, "target") else dataset.targets
    label_idx_split: Dict = {}
    for i, _ in enumerate(label):
        label_i = label[i].item()
        if label_i not in label_idx_split:
            label_idx_split[label_i] = []
        label_idx_split[label_i].append(i)

    shard_per_class = int(shard_per_user * num_clients / classes_size)

    for label_i in label_idx_split:
        label_idx = label_idx_split[label_i]
        num_leftover = len(label_idx) % shard_per_class
        leftover = label_idx[-num_leftover:] if num_leftover > 0 else []
        new_label_idx = (
            np.array(label_idx[:-num_leftover])
            if num_leftover > 0
            else np.array(label_idx)
        )
        new_label_idx = new_label_idx.reshape((shard_per_class, -1)).tolist()

        for i, leftover_label_idx in enumerate(leftover):
            new_label_idx[i] = np.concatenate([new_label_idx[i], [leftover_label_idx]])
        label_idx_split[label_i] = new_label_idx
    return label_idx_split, shard_per_class


def _get_dataset_from_idx(dataset, data_split, num_clients):
    divided_dataset = [None for i in range(num_clients)]
    for i in range(num_clients):
        divided_dataset[i] = Subset(dataset, data_split[i])
    return divided_dataset


def _balance_classes(
    trainset: Dataset,
    seed: Optional[int] = 42,
) -> Dataset:
    class_counts = np.bincount(trainset.target)
    targets = torch.Tensor(trainset.target)
    smallest = np.min(class_counts)
    idxs = targets.argsort()
    tmp = [Subset(trainset, idxs[: int(smallest)])]
    tmp_targets = [targets[idxs[: int(smallest)]]]
    for count in np.cumsum(class_counts):
        tmp.append(Subset(trainset, idxs[int(count) : int(count + smallest)]))
        tmp_targets.append(targets[idxs[int(count) : int(count + smallest)]])
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


# pylint: disable=too-many-locals, too-many-arguments
def _power_law_split(
    sorted_trainset: Dataset,
    num_partitions: int,
    num_labels_per_partition: int = 2,
    min_data_per_partition: int = 10,
    mean: float = 0.0,
    sigma: float = 2.0,
) -> Dataset:
    """Partition the dataset following a power-law distribution. It follows the.

    implementation of Li et al 2020: https://arxiv.org/abs/1812.06127 with default
    values set accordingly.

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
    full_idx = list(range(len(targets)))

    class_counts = np.bincount(sorted_trainset.targets)
    labels_cs = np.cumsum(class_counts)
    labels_cs = [0] + labels_cs[:-1].tolist()

    partitions_idx: List[List[int]] = []
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
    # obtain how many samples each partition should be assigned for each of the
    # labels it contains
    # pylint: disable=too-many-function-args
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
