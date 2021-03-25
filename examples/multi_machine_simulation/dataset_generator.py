from typing import Tuple, List
from pathlib import Path

import tensorflow as tf
import numpy as np

XY = Tuple[np.ndarray, np.ndarray]


def shuffle(x: np.ndarray, y: np.ndarray) -> XY:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx]


def partition(x: np.ndarray, y: np.ndarray, num_partitions: int) -> List[XY]:
    """Return x, y as list of partitions."""
    return list(zip(np.split(x, num_partitions), np.split(y, num_partitions)))


def create_partitions(num_partitions: int):
    Path("./partitions").mkdir(exist_ok=True)

    # Load dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Shuffle
    (x_train, y_train) = shuffle(x_train, y_train)
    (x_test, y_test) = shuffle(x_test, y_test)

    # Partition
    xy_train = partition(x_train, y_train, num_partitions)
    xy_test = partition(x_test, y_test, num_partitions)

    # Store train partitons
    for index, xy_part in enumerate(xy_train):
        (x, y) = xy_part
        np.save(f"./partitions/x_train_{index}.npy", x)
        np.save(f"./partitions/y_train_{index}.npy", y)

    # Store test partitions
    for index, xy_part in enumerate(xy_test):
        (x, y) = xy_part
        np.save(f"./partitions/x_test_{index}.npy", x)
        np.save(f"./partitions/y_test_{index}.npy", y)

if __name__ == "__main__":
    create_partitions(num_partitions=10)
