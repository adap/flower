"""$project_name: A Flower / JAX app."""

import flwr as fl

# Configure the strategy
strategy = fl.server.strategy.FedAvg()

# Flower ServerApp
app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
