"""custom_metrics_example: A Flower app for custom metrics."""

from typing import Any, List

import tensorflow as tf
from flwr_datasets import FederatedDataset


def load_data(partition_id, num_partitions) -> tuple[List, List, List, List]:
    """Load data with Flower Datasets (CIFAR-10)."""
    fds = FederatedDataset(dataset="cifar10", partitioners={"train": num_partitions})
    train = fds.load_split("train")
    test = fds.load_split("test")

    # Using Numpy format
    train_np = train.with_format("numpy")
    test_np = test.with_format("numpy")
    x_train, y_train = train_np["img"], train_np["label"]
    x_test, y_test = test_np["img"], test_np["label"]

    return x_train, y_train, x_test, y_test


def get_model(width, height, num_channels) -> Any:
    """Load model (MobileNetV2)."""
    model = tf.keras.applications.MobileNetV2(
        (width, height, num_channels),
        classes=10,
        weights=None,
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
