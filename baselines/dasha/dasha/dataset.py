"""dasha: A Flower Baseline."""

from enum import Enum

import numpy as np
import torch
import torch.utils.data as data_utils
import torchvision
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset
from torchvision import transforms

from dasha.dataset_preparation import (
    DATA_PATH,
    DatasetType,
    train_dataset_path,
)


class LIBSVMDatasetName(Enum):
    """Enum for the LibSVM datasets."""

    MUSHROOMS = "mushrooms"


def _load_libsvm_dataset(cfg) -> Dataset:
    assert cfg["dataset"]["type"] == DatasetType.LIBSVM.value
    path_to_dataset = DATA_PATH
    dataset_name = cfg["dataset"]["name"]
    # pylint: disable=unbalanced-tuple-unpacking
    data, labels = load_svmlight_file(
        train_dataset_path(path_to_dataset, dataset_name)
    )
    data = data.toarray().astype(np.float32)
    print_labels = np.unique(labels, return_counts=True)
    data_shape = data.shape
    print(f"Original labels: {print_labels}")
    print(f"Features Shape: {data_shape}")
    if dataset_name == LIBSVMDatasetName.MUSHROOMS.value:
        labels = labels.astype(np.int64)
        remap_labels = np.zeros_like(labels)
        remap_labels[labels == 1] = 0
        remap_labels[labels != 1] = 1
        labels = remap_labels
    else:
        raise RuntimeError("Wrong dataset")
    dataset = data_utils.TensorDataset(
        torch.Tensor(data), torch.Tensor(labels)
    )
    return dataset


def _load_test_dataset(cfg) -> Dataset:
    assert cfg.dataset.type == DatasetType.TEST.value
    features = [[1], [2]]
    targets = [[1], [2]]
    dataset = data_utils.TensorDataset(
        torch.Tensor(features), torch.Tensor(targets)
    )
    return dataset


def _load_random_test_dataset() -> Dataset:
    generator = np.random.default_rng(42)
    features = np.concatenate(
        ((1 + generator.normal(size=100)), (1.1 + generator.normal(size=100)))
    ).reshape(-1, 1)
    targets = np.concatenate((torch.zeros(100), torch.ones(100))).reshape(
        -1, 1
    )
    dataset = data_utils.TensorDataset(
        torch.Tensor(features), torch.Tensor(targets)
    )
    return dataset


def _load_cifar10() -> Dataset:
    transform_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    path_to_dataset = DATA_PATH
    trainset = torchvision.datasets.CIFAR10(
        root=path_to_dataset,
        train=True,
        download=False,
        transform=transform_train,
    )
    return trainset


def load_dataset(cfg) -> Dataset:
    """Load a dataset."""
    if cfg["dataset"]["type"] == DatasetType.LIBSVM.value:
        return _load_libsvm_dataset(cfg)
    if cfg["dataset"]["type"] == DatasetType.CIFAR10.value:
        return _load_cifar10()
    if cfg["dataset"]["type"] == DatasetType.TEST.value:
        return _load_test_dataset(cfg)
    if cfg["dataset"]["type"] == DatasetType.RANDOM_TEST.value:
        return _load_random_test_dataset()
    raise RuntimeError("Wrong dataset type")


def random_split(dataset, num_clients, seed=42):
    """Split randomly a dataset."""
    lengths = [1 / num_clients] * num_clients
    datasets = torch.utils.data.random_split(
        dataset, lengths, torch.Generator().manual_seed(seed)
    )
    return datasets
