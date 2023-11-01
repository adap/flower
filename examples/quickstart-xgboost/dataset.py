import numpy as np
import datasets
import xgboost as xgb

from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (IidPartitioner, LinearPartitioner,
                                       SquarePartitioner, ExponentialPartitioner)

SPLIT_DICT = {"uniform": IidPartitioner,
              "linear": LinearPartitioner,
              "square": SquarePartitioner,
              "exponential": ExponentialPartitioner
              }


def init_higgs(num_partitions: int, split_method: str) -> FederatedDataset:
    """Initialise FederatedDataset based on selected split method."""
    partitioner = SPLIT_DICT[split_method](num_partitions=num_partitions)
    fds = FederatedDataset(dataset="jxie/higgs", partitioners={"train": partitioner})
    return fds


def load_partition(fds: FederatedDataset, partition_id: int) -> datasets.Dataset:
    """Load partition based on the given partition ID."""
    partition = fds.load_partition(idx=partition_id, split="train")
    partition.set_format("numpy")
    return partition


def split_train_test(partition: datasets.Dataset, split_rate: float, seed: int):
    """Split the data into train and validation set given split rate."""
    train_test = partition.train_test_split(test_size=split_rate, seed=seed)
    partition_train = train_test["train"]
    partition_test = train_test["test"]

    # Reformat data for xgboost input
    train_data = _reformat_data(partition_train)
    val_data = _reformat_data(partition_test)
    return train_data, val_data


def _reformat_data(partition):
    x = partition["inputs"]
    y = partition["label"]
    new_data = xgb.DMatrix(x, label=y)
    return new_data




