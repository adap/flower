"""Dataset utilities for federated learning."""

import numpy as np
from tensorflow import keras
from keras.utils import to_categorical

from fedavgm.common import create_lda_partitions


def prepare_dataset(fmnist):
    """Prepare the dataset.

    This method considers FMNIST or CIFAR-10 for creating both train and test sets. The
    sets are already normalized.
    """
    if fmnist is True:
        print(">>> [Dataset] Loading FMNIST...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        input_shape = x_train.shape[1:]
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
    else:
        print(">>> [Dataset] Loading CIFAR-10...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255
        input_shape = x_train.shape[1:]
        num_classes = len(np.unique(y_train))
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)
        
    return x_train, y_train, x_test, y_test, input_shape, num_classes


def partition(x_train, y_train, num_clients, concentration, num_classes):
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
        concentration=concentration * num_classes,
        seed=1234,
    )
    return partitions
