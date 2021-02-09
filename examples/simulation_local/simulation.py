import os
import time
from multiprocessing import Process
from typing import Tuple

import flwr as fl
import numpy as np
import tensorflow as tf

from flwr.server.server import Server
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager
from flwr.server.network_manager import SimpleInMemoryNetworkManager


import dataset

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


class CifarClient(fl.client.NumPyClient):
    def __init__(self, dataset: DATASET) -> None:
        super().__init__()
        self.dataset = dataset

    def get_parameters(self):
        """Return current weights."""
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        return model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training
        examples."""
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        model.set_weights(parameters)

        (x_train, y_train), _ = self.dataset

        model.fit(x_train, y_train, epochs=1, batch_size=32)

        weights = model.get_weights()

        return weights, len(x_train)

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
        model.set_weights(parameters)

        _, (x_test, y_test) = self.dataset

        loss, accuracy = model.evaluate(x_test, y_test)

        return len(x_test), loss, accuracy


def run_simulation(num_rounds: int, num_clients: int):
    """Start a FL simulation."""

    # Load the dataset partitions
    partitions = dataset.load(num_partitions=num_clients)
    clients = [CifarClient(dataset=partition) for partition in partitions]

    # Use custom in memory network manager instead of default gRPC one
    network_manager = SimpleInMemoryNetworkManager(clients=clients)

    fl.server.start_server(
        network_managers=[network_manager],
        config={"num_rounds": num_rounds},
    )


if __name__ == "__main__":
    run_simulation(num_rounds=10, num_clients=10)
