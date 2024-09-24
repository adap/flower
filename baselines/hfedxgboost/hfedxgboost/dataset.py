"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from typing import List, Optional, Tuple, Union

import torch
from flwr.common import NDArray
from torch.utils.data import DataLoader, Dataset, random_split

from hfedxgboost.dataset_preparation import (
    datafiles_fusion,
    download_data,
    modify_labels,
    train_test_split,
)


def load_single_dataset(
    task_type: str, dataset_name: str, train_ratio: Optional[float] = 0.75
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Load a single dataset.

    Parameters
    ----------
        task_type (str): The type of task, either "BINARY" or "REG".
        dataset_name (str): The name of the dataset to load.
        train_ratio (float, optional): The ratio of training data to the total dataset.
        Default is 0.75.

    Returns
    -------
            x_train (numpy array): The training data features.
            y_train (numpy array): The training data labels.
            X_test (numpy array): The testing data features.
            y_test (numpy array): The testing data labels.
    """
    datafiles_paths = download_data(dataset_name)
    X, Y = datafiles_fusion(datafiles_paths)
    x_train, y_train, x_test, y_test = train_test_split(X, Y, train_ratio=train_ratio)
    if task_type.upper() == "BINARY":
        y_train, y_test = modify_labels(y_train, y_test)

        print(
            "First class ratio in train data",
            y_train[y_train == 0.0].size / x_train.shape[0],
        )
        print(
            "Second class ratio in train data",
            y_train[y_train != 0.0].size / x_train.shape[0],
        )
        print(
            "First class ratio in test data",
            y_test[y_test == 0.0].size / x_test.shape[0],
        )
        print(
            "Second class ratio in test data",
            y_test[y_test != 0.0].size / x_test.shape[0],
        )

    print("Feature dimension of the dataset:", x_train.shape[1])
    print("Size of the trainset:", x_train.shape[0])
    print("Size of the testset:", x_test.shape[0])

    return x_train, y_train, x_test, y_test


def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    """Return a DataLoader object.

    Parameters
    ----------
        dataset (Dataset): The dataset object that contains the data.
        partition (str): The partition string that specifies the subset of data to use.
        batch_size (Union[int, str]): The batch size to use for loading data.
        It can be either an integer value or the string "whole".
        If "whole" is provided, the batch size will be set to the length of the dataset.

    Returns
    -------
        DataLoader: A DataLoader object that loads data from the dataset in batches.
    """
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )


def divide_dataset_between_clients(
    trainset: Dataset,
    testset: Dataset,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
) -> Tuple[DataLoader, Union[List[DataLoader], List[None]], DataLoader]:
    """Divide the data between clients with IID distribution.

    Parameters
    ----------
        trainset (Dataset): The  full training dataset.
        testset (Dataset): The full test dataset.
        pool_size (int): The number of partitions to create.
        batch_size (Union[int, str]): The size of the batches.
        val_ratio (float, optional): The ratio of validation data. Defaults to 0.0.

    Returns
    -------
        Tuple[DataLoader, DataLoader, DataLoader]: A tuple containing
        the training loaders, validation loaders (or None), and test loader.
    """
    # Split training set into `num_clients` partitions to simulate
    # different local datasets
    trainset_length = len(trainset)
    lengths = [trainset_length // pool_size] * pool_size
    if sum(lengths) != trainset_length:
        lengths[-1] = trainset_length - sum(lengths[0:-1])
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))

    # Split each partition into train/val and create DataLoader
    trainloaders: List[DataLoader] = []
    valloaders: Union[List[DataLoader], List[None]] = []
    for dataset in datasets:
        len_val = int(len(dataset) * val_ratio)
        len_train = len(dataset) - len_val
        ds_train, ds_val = random_split(
            dataset, [len_train, len_val], torch.Generator().manual_seed(0)
        )
        trainloaders.append(get_dataloader(ds_train, "train", batch_size))
        if len_val != 0:
            valloaders.append(get_dataloader(ds_val, "val", batch_size))
        else:
            valloaders.append(None)
    return trainloaders, valloaders, get_dataloader(testset, "test", batch_size)
