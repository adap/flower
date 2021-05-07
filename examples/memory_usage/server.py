import os

import flwr as fl
from flwr.server.strategy import FedAvg

# Make TensorFlow log less verbose
os.environ["TF_C PP_MIN_LOG_LEVEL"] = "3"

def start_server(num_rounds: int, num_clients: int, fraction_fit: float):
    """Start the server with a slightly adjusted FedAvg strategy."""
    strategy = FedAvg(min_available_clients=num_clients, fraction_fit=fraction_fit)
    fl.server.start_server(strategy=strategy, config={"num_rounds": num_rounds})

if __name__ == "__main__":
    start_server(num_rounds=5000, num_clients=100, fraction_fit=0.5)
