from multiprocessing import Process
import os
from typing import Tuple

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time
import tensorflow as tf
import numpy as np
import flwr as fl
from flwr.server.strategy import FedAvg

import dataset

DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def start_server(num_clients: int, num_rounds: int):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=0.5)
    fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds})


def start_client(dataset: DATASET) -> None:
    """Start a single client with the provided dataset."""

    # Load and compile a Keras model for CIFAR-10
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Unpack the CIFAR-10 dataset partition
    (x_train, y_train), (x_test, y_test) = dataset

    # Define a Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            return model.get_weights()

        def fit(self, parameters, config):
            model.set_weights(parameters)
            model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
            return model.get_weights(), len(x_train)

        def evaluate(self, parameters, config):
            model.set_weights(parameters)
            loss, accuracy = model.evaluate(x_test, y_test)
            return len(x_test), loss, accuracy

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=CifarClient())


def run_simulation(num_clients: int, num_rounds: int):
    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(target=start_server, args=(num_clients, num_rounds))
    server_process.start()
    processes.append(server_process)

    # Load the dataset partitions
    partitions = dataset.load(num_partitions=num_clients)

    # Start all the clients
    for partition in partitions:
        client_process = Process(target=start_client, args=(partition,))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    run_simulation(num_clients=10, num_rounds=200)
