import os
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server.utils import tensorboard

# Directory named flwr_logs besides this server.py file
LOGDIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "flwr_logs")


if __name__ == "__main__":
    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=tensorboard(logdir=LOGDIR)(FedAvg)(),
        config={"num_rounds": 3},
    )
