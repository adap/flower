"""$project_name: A Flower / Scikit-Learn app."""

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg


strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,
)

# Create ServerApp
app = ServerApp(
    config=ServerConfig(num_rounds=1),
    strategy=strategy,
)
