"""Preprocess a dataset."""

import os
import pathlib
import urllib.request
from enum import Enum
from typing import Dict

import torchvision
from omegaconf import DictConfig


class DatasetType(Enum):
    """Enum for types of datasets."""

    LIBSVM = "libsvm"
    CIFAR10 = "cifar10"
    TEST = "test"
    RANDOM_TEST = "random_test"


class _DatasetSplit(Enum):
    TRAIN = "train"


def train_dataset_path(path_to_dataset, dataset_name):
    """Prepare a path to a dataset."""
    return os.path.join(
        path_to_dataset, "{}_{}".format(dataset_name, _DatasetSplit.TRAIN.value)
    )


def _prepare_libsvm(
    path_to_dataset: str, dataset_name: str, dataset_urls: Dict[str, Dict[str, str]]
) -> None:
    assert path_to_dataset is not None
    target_file = train_dataset_path(path_to_dataset, dataset_name)
    if os.path.exists(target_file):
        return
    print("Downloading the dataset")
    dataset_url = dataset_urls[dataset_name][_DatasetSplit.TRAIN.value]
    urllib.request.urlretrieve(dataset_url, target_file)


def _prepare_cifar10(path_to_dataset: str) -> None:
    torchvision.datasets.CIFAR10(root=path_to_dataset, train=True, download=True)


def find_pre_downloaded_or_download_dataset(cfg: DictConfig) -> None:
    """Find a pre-downloaded dataset or downloads it.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    if cfg.dataset.path_to_dataset is None:
        current_file_path = pathlib.Path(__file__).parent.resolve()
        path_to_dataset = os.path.join(current_file_path, "conf", "dataset")
        print(
            f"The parameter cfg.dataset.path_to_dataset is not specified. \
                We will use the default path {path_to_dataset}."
        )
        cfg.dataset.path_to_dataset = path_to_dataset
    else:
        assert os.path.isdir(
            cfg.dataset.path_to_dataset
        ), f"The folder {cfg.dataset.path_to_dataset} does not exists"
    if cfg.dataset.type == DatasetType.LIBSVM.value:
        _prepare_libsvm(
            cfg.dataset.path_to_dataset,
            cfg.dataset.dataset_name,
            cfg.dataset._dataset_urls,  # pylint: disable=protected-access
        )
    elif cfg.dataset.type == DatasetType.CIFAR10.value:
        _prepare_cifar10(cfg.dataset.path_to_dataset)
    else:
        raise RuntimeError()
