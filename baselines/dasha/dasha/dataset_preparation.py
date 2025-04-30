"""dasha: A Flower Baseline."""

import os
import pathlib
import urllib.request
from enum import Enum

import torchvision

DATA_PATH = pathlib.Path(os.path.abspath(__file__)).parent.parent / "data"


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
        path_to_dataset, f"{dataset_name}_{_DatasetSplit.TRAIN.value}"
    )


def _prepare_libsvm(
    path_to_dataset: pathlib.Path, dataset_name: str, train_url: str
) -> None:
    assert path_to_dataset is not None
    target_file = train_dataset_path(path_to_dataset, dataset_name)
    if os.path.exists(target_file):
        return
    pathlib.Path(path_to_dataset).mkdir(parents=True, exist_ok=True)
    print("Downloading the dataset")
    dataset_url = train_url
    urllib.request.urlretrieve(dataset_url, target_file)


def _prepare_cifar10(path_to_dataset: pathlib.Path) -> None:
    torchvision.datasets.CIFAR10(
        root=path_to_dataset, train=True, download=True
    )


def find_pre_downloaded_or_download_dataset(cfg) -> None:
    """Find a pre-downloaded dataset or downloads it.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    if cfg["dataset"]["type"] == DatasetType.LIBSVM.value:
        _prepare_libsvm(
            DATA_PATH,
            cfg["dataset"]["name"],
            cfg["dataset"]["train-url"],  # pylint: disable=protected-access
        )
    elif cfg["dataset"]["type"] == DatasetType.CIFAR10.value:
        _prepare_cifar10(DATA_PATH)
    else:
        raise RuntimeError()
