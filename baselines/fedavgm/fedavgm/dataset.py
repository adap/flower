"""Dataset utilities for federated learning."""

import numpy as np
from tensorflow import keras

from fedavgm.common import create_lda_partitions


def cifar10(num_classes, input_shape):
    """Prepare the CIFAR-10.

    This method considers CIFAR-10 for creating both train and test sets. The sets are
    already normalized.
    """
    print(f">>> [Dataset] Loading CIFAR-10. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def fmnist(num_classes, input_shape):
    """Prepare the FMNIST.

    This method considers FMNIST for creating both train and test sets. The sets are
    already normalized.
    """
    print(f">>> [Dataset] Loading FMNIST. {num_classes} | {input_shape}.")
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    return x_train, y_train, x_test, y_test, input_shape, num_classes


def partition(x_train, y_train, num_clients, concentration):
    """Create non-iid partitions.

    The partitions uses a LDA distribution based on concentration.
    """
    print(
        f">>> [Dataset] {num_clients} clients, non-iid concentration {concentration}..."
    )
    dataset = [x_train, y_train]
    partitions, _ = create_lda_partitions(
        dataset,
        num_partitions=num_clients,
        # concentration=concentration * num_classes,
        concentration=concentration,
        seed=1234,
    )
    return partitions
