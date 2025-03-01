"""statavg: A Flower Baseline."""

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split


def get_split_dataset(
    path_to_dataset: str, include_testset: bool, testset_ratio: float
) -> Tuple[DataFrame, DataFrame]:
    """Load and split datsaet to train (client) and test (server)."""
    script_dir = Path(os.path.abspath(__file__)).parent.parent
    dataset_path = os.path.join(script_dir, path_to_dataset, "dataset.csv")
    dataset = pd.read_csv(dataset_path)

    # remove NaN
    dataset = dataset.dropna(axis=0, how="any")

    # keep label
    Y = dataset[["type"]]

    # remove irrelevant features
    dataset = dataset.drop(["PID", "CMD", "label"], axis=1)

    # stratified split
    if include_testset:
        trainset, testset, _, _ = train_test_split(
            dataset, Y, test_size=testset_ratio, stratify=Y, random_state=41
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
    x_curr = X
    y_curr = Y
    partition_dataset = []
    for _ in range(num_partitions - 1):
        x_curr, x_temp, y_curr, _ = train_test_split(
            x_curr,
            y_curr,
            test_size=1 / num_partitions,
            stratify=y_curr,
            random_state=41,
        )
        partition_dataset.append(x_temp)
        num_partitions = num_partitions - 1
    partition_dataset.append(x_curr)
    return partition_dataset


def prepare_dataset(
    num_partitions: int,
    path_to_dataset: str,
    include_testset: bool,
    testset_ratio: float,
) -> Tuple[DataFrame, List[DataFrame]]:
    """Create the following partitions.

    train_partitions: trainsets for clients.
    testset: testset for server (if server-side evaluation is needed).
    """
    trainset, testset = get_split_dataset(
        path_to_dataset, include_testset, testset_ratio
    )
    y_trainset = trainset["type"]
    train_partitions = split_clients_data(trainset, y_trainset, num_partitions)

    return train_partitions, testset
