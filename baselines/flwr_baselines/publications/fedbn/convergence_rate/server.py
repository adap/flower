"""Flower server example."""

import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )
    fl.server.start_server(
        server_address="[::]:8000",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )
