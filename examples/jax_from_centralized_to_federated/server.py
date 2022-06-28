"""Flower server example."""

import flwr as fl

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config={"num_rounds": 3},
    )
