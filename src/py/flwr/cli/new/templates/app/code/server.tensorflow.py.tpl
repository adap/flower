"""$project_name: A Flower / TensorFlow app."""

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

# Define config
config = ServerConfig(num_rounds=3)

strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,
)

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)
