import os
from multiprocessing import Process

import flwr as fl

from model import FAKE_MOBILENET_V2

# Make TensorFlow log less verbose
os.environ["TF_C PP_MIN_LOG_LEVEL"] = "3"

# Define a Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        """Return current weights."""
        return FAKE_MOBILENET_V2

    def fit(self, parameters, config):
        """Fit model and return new weights as well as number of training
        examples."""
        return FAKE_MOBILENET_V2, 1, {}

    def evaluate(self, parameters, config):
        """Evaluate using provided parameters."""
        return 0.0, 1, {"accuracy": 1}


def start_client() -> None:
    """Start a single client with the provided dataset."""
    # Start Flower client
    fl.client.start_numpy_client("0.0.0.0:8080", client=CifarClient())


def run_clients(num_clients: int):
    """Start a FL simulation."""

    # This will hold all the processes which we are going to create
    processes = []

    # Start all the clients
    for _ in range(num_clients):
        client_process = Process(target=start_client)
        client_process.start()
        processes.append(client_process)

    # Block until all processes are finished
    for p in processes:
        p.join()


if __name__ == "__main__":
    run_clients(num_clients=50)
