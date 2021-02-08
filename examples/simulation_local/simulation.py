import os
import time
from multiprocessing import Process
from typing import Tuple

import flwr as fl
from flwr.client.numpy_client import NumPyClientWrapper
import numpy as np
from numpy.core.fromnumeric import partition
import tensorflow as tf
from flwr.server.server import Server
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import SimpleClientManager
from flwr.server.network_manager import SimpleInMemoryNetworkManager


import dataset

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def generate_client(dataset: DATASET) -> None:
    """Start a single client with the provided dataset."""

    # Load and compile a Keras model for CIFAR-10
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Unpack the CIFAR-10 dataset partition
    (x_train, y_train), (x_test, y_test) = dataset

    # Define a Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            """Return current weights."""
            return model.get_weights()

        def fit(self, parameters, config):
            """Fit model and return new weights as well as number of training
            examples."""
            model.set_weights(parameters)
            # Remove steps_per_epoch if you want to train over the full dataset
            # https://keras.io/api/models/model_training_apis/#fit-method
            model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
            return model.get_weights(), len(x_train)

        def evaluate(self, parameters, config):
            """Evaluate using provided parameters."""
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return len(x_test), loss, accuracy

    return CifarClient()


def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start a FL simulation."""

    # Load the dataset partitions
    partitions = dataset.load(num_partitions=num_clients)
    clients = [generate_client(partition) for partition in partitions]

    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
    client_manager = SimpleClientManager()
    server = Server(client_manager=client_manager, strategy=strategy)

    # Use custom in memory network manager instead of default gRPC one
    network_manager = SimpleInMemoryNetworkManager(
        client_manager=server.client_manager(), clients=clients
    )

    fl.server.start_server(
        server=server,
        network_managers=[network_manager],
        config={"num_rounds": num_rounds},
    )


if __name__ == "__main__":
    run_simulation(num_rounds=2, num_clients=5, fraction_fit=0.4)
