"""Flower server example."""


import flwr as fl

if __name__ == "__main__":
    fl.server.start_server("localhost:9080", config={"num_rounds": 30})
