"""Flower server example."""


import flwr as fl

if __name__ == "__main__":
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,
        min_eval_clients=2,
        min_available_clients=2,
    )
    fl.server.start_server("[::]:8080", config={"num_rounds": 100}, strategy=strategy)
