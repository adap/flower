import flwr as fl
from flwr.server.strategy import FedAvg

import mlcube_utils as mlcube


def main():
    strategy = FedAvg(initial_parameters=mlcube.initial_parameters())
    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=3),
    )


if __name__ == "__main__":
    main()
