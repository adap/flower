"""huggingface_example: A Flower / Hugging Face app."""

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
)

config = ServerConfig(num_rounds=3)

app = ServerApp(
    config=config,
    strategy=strategy,
)
