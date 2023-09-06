"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""
from dataset_preparation import download_data,datafiles_fusion,train_test_split,modify_labels
from flwr.common import NDArray, NDArrays
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset, random_split

def load_single_dataset(task_type,dataset_name,train_ratio=.75):
    datafiles_paths=download_data(dataset_name)
    X,Y=datafiles_fusion(datafiles_paths)
    X_train,y_train,X_test,y_test=train_test_split(X,Y,train_ratio=train_ratio)
    if task_type.upper()=="BINARY":
        y_train,y_test=modify_labels(y_train,y_test)
    return X_train,y_train,X_test,y_test


def get_dataloader(
    dataset: Dataset, partition: str, batch_size: Union[int, str]
) -> DataLoader:
    if batch_size == "whole":
        batch_size = len(dataset)
    return DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, shuffle=(partition == "train")
    )


def do_fl_partitioning(
    trainset: Dataset,
    testset: Dataset,
    pool_size: int,
    batch_size: Union[int, str],
    val_ratio: float = 0.0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Split training set into `num_clients` partitions to simulate different local datasets
    trainset_length=len(trainset)
    partition_size = trainset_length // pool_size
    lengths = [partition_size] * pool_size
    if sum(lengths) != trainset_length:
        lengths[-1] = trainset_length - sum(lengths[0:-1])
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(0))

    # Split each partition into train/val and create DataLoader
    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = int(len(ds) * val_ratio)
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(0))
        trainloaders.append(get_dataloader(ds_train, "train", batch_size))
        if len_val != 0:
            valloaders.append(get_dataloader(ds_val, "val", batch_size))
        else:
            valloaders = None
    testloader = get_dataloader(testset, "test", batch_size)
    return trainloaders, valloaders, testloader