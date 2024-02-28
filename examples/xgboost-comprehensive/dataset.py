import xgboost as xgb
from typing import Union
from datasets import Dataset, DatasetDict, concatenate_datasets
from flwr_datasets.partitioner import (
    IidPartitioner,
    LinearPartitioner,
    SquarePartitioner,
    ExponentialPartitioner,
)

CORRELATION_TO_PARTITIONER = {
    "uniform": IidPartitioner,
    "linear": LinearPartitioner,
    "square": SquarePartitioner,
    "exponential": ExponentialPartitioner,
}


def instantiate_partitioner(partitioner_type: str, num_partitions: int):
    """Initialise partitioner based on selected partitioner type and number of
    partitions."""
    partitioner = CORRELATION_TO_PARTITIONER[partitioner_type](
        num_partitions=num_partitions
    )
    return partitioner


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
    x, y = separate_xy(data)
    new_data = xgb.DMatrix(x, label=y)
    return new_data


def separate_xy(data: Union[Dataset, DatasetDict]):
    """Return outputs of x (data) and y (labels) ."""
    x = data["inputs"]
    y = data["label"]
    return x, y


def resplit(dataset: DatasetDict) -> DatasetDict:
    """Increase the quantity of centralised test samples from 500K to 1M."""
    return DatasetDict(
        {
            "train": dataset["train"].select(
                range(0, dataset["train"].num_rows - 500_000)
            ),
            "test": concatenate_datasets(
                [
                    dataset["train"].select(
                        range(
                            dataset["train"].num_rows - 500_000,
                            dataset["train"].num_rows,
                        )
                    ),
                    dataset["test"],
                ]
            ),
        }
    )
