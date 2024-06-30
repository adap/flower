"""mlxexample: A Flower / MLX app."""

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg


# Define strategy
strategy = FedAvg()


# Create ServerApp
app = ServerApp(
    config=ServerConfig(num_rounds=3),
    strategy=strategy,
)
