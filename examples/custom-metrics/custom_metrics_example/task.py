"""custom_metrics_example: A Flower app for custom metrics."""

from typing import Any

import numpy as np
import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner

fds = None  # Cache FederatedDataset


def load_data(
    partition_id: int, num_partitions: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load data with Flower Datasets (CIFAR-10)."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2, seed=42)
    x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]

    return x_train, y_train, x_test, y_test


def get_model(width: int = 32, height: int = 32, num_channels: int = 3) -> Any:
    """Load model (MobileNetV2)."""
    model = tf.keras.applications.MobileNetV2(
        (width, height, num_channels),
        classes=10,
        weights=None,
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
