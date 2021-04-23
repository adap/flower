import os
import time
from multiprocessing import Process
import threading
from typing import Optional, Tuple
import resource

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.server.strategy import FedAvg

import dataset

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

DATASET = Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

partitions: Optional[dataset.PartitionedDataset] = None

# Define a Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, partition_id) -> None:
        self.model = model
        self.partition_id = partition_id

    def get_parameters(self):
        """Return current weights."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training
        examples."""
        self.model.set_weights(parameters)
        (x_train, y_train), _ = partitions[self.partition_id]
        # Remove steps_per_epoch if you want to train over the full dataset
        # https://keras.io/api/models/model_training_apis/#fit-method
        self.model.fit(x_train, y_train, epochs=1, batch_size=32, steps_per_epoch=3)
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        self.model.set_weights(parameters)
        _, (x_test, y_test) = partitions[self.partition_id]
        loss, accuracy = self.model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}

def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
    # Exposes the server by default on port 8080
    fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds})


def start_client(partition_id: int) -> None:
    """Start a single client with the provided dataset."""
    model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=CifarClient(model, partition_id))


def run_simulation(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start the server
    server_process = Process(
        target=start_server, args=(num_rounds, num_clients, fraction_fit)
    )
    server_process.start()
    processes.append(server_process)

    # Optionally block the script here for a second or two so the server has time to start
    time.sleep(2)

    # Start all the clients
    for partition_id in range(len(partitions)):
        client_process = Process(target=start_client, args=(partition_id,))
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()

if __name__ == "__main__":
    num_clients = 10
    partitions = dataset.load(num_partitions=num_clients)
    run_simulation(num_rounds=1000, num_clients=num_clients, fraction_fit=0.5)
