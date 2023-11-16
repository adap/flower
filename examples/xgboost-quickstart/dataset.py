import xgboost as xgb
from typing import Union
from datasets import Dataset, DatasetDict


def train_test_split(partition: Dataset, test_fraction: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=test_fraction, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    num_train = len(partition_train)
    num_test = len(partition_test)

    return partition_train, partition_test, num_train, num_test


def transform_dataset_to_dmatrix(data: Union[Dataset, DatasetDict]) -> xgb.core.DMatrix:
    """Transform dataset to DMatrix format for xgboost."""
    x = data["inputs"]
    y = data["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data
