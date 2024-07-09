"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

# Define strategy
strategy = FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=0.5,
)

# Define training config
config = ServerConfig(num_rounds=3)

app = ServerApp(
    config=config,
    strategy=strategy,
)
