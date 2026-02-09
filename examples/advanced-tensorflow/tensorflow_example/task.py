"""tensorflow-example: A Flower / TensorFlow app."""

import json
import os
from datetime import datetime
from pathlib import Path

import keras
from flwr.common.typing import UserConfig
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from keras import layers

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for FashionMNIST and set Adam optimizer
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    """Load partition FashionMNIST data."""
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=1.0,
            seed=42,
        )
        fds = FederatedDataset(
            dataset="zalando-datasets/fashion_mnist",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id, "train")
    partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    partition = partition.train_test_split(test_size=0.2)

    partition["train"].set_format(type="numpy", columns=["image", "label"])
    partition["test"].set_format(type="numpy", columns=["image", "label"])

    x_train = partition["train"][:]["image"].astype("float32") / 255.0
    y_train = partition["train"][:]["label"]
    x_test = partition["test"][:]["image"].astype("float32") / 255.0
    y_test = partition["test"][:]["label"]

    return x_train, y_train, x_test, y_test


def create_run_dir(config: UserConfig) -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)

    # Save run config as json
    with open(f"{save_path}/run_config.json", "w", encoding="utf-8") as fp:
        json.dump(config, fp)

    return save_path, run_dir
