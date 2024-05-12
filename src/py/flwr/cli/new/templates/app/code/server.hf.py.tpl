"""$project_name: A Flower / HuggingFace Transformers app."""

from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig


# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

# Start server
app = ServerApp(
    config=ServerConfig(num_rounds=3),
    strategy=strategy,
)
