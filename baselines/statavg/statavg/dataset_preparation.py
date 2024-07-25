"""Prepare the dataset."""

import os
from typing import List, Tuple

import pandas as pd
from omegaconf import DictConfig
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def get_split_dataset(
    path_to_dataset: str, include_testset: DictConfig
) -> Tuple[DataFrame, DataFrame]:
    """Load and split datsaet to train (client) and test (server)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "dataset", "dataset.csv")
    dataset = pd.read_csv(dataset_path)

    # remove NaN
    dataset = dataset.dropna(axis=0, how="any")

    # keep label
    Y = dataset[["type"]]

    # remove irrelevant features
    dataset = dataset.drop(["ts", "PID", "CMD", "label"], axis=1)

    # stratified split
    if include_testset.flag:
        trainset, testset, Y_train, Y_test = train_test_split(
            dataset, Y, test_size=include_testset.ratio, stratify=Y, random_state=41
        )
        res = trainset, testset
    else:
        # if include_test is false, return an empty DataFrame for the testset
        res = dataset, pd.DataFrame()

    return res


def split_clients_data(
    X: DataFrame, Y: DataFrame, num_partitions: int
) -> List[DataFrame]:
    """Perform stratified split based on labels."""
    X_curr = X
    Y_curr = Y
    partition_dataset = []
    for _ in range(num_partitions - 1):
        X_curr, X_temp, Y_curr, Y_temp = train_test_split(
            X_curr,
            Y_curr,
            test_size=1 / num_partitions,
            stratify=Y_curr,
            random_state=41,
        )
        partition_dataset.append(X_temp)
        num_partitions = num_partitions - 1
    partition_dataset.append(X_curr)
    return partition_dataset


def prepare_dataset(
    num_partitions: int, path_to_dataset: str, include_testset: DictConfig
) -> Tuple[DataFrame, List[DataFrame]]:
    """Create the following partitions.

    train_partitions: trainsets for clients.
    testset: testset for server (if server-side evaluation is needed).
    """
    trainset, testset = get_split_dataset(path_to_dataset, include_testset)
    y_trainset = trainset["type"]
    train_partitions = split_clients_data(trainset, y_trainset, num_partitions)

    return train_partitions, testset
