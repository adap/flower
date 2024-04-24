"""$project_name: A Flower / PyTorch app."""

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from $project_name.task import Net, get_weights


# Initialize model parameters
ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)


# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,
    initial_parameters=parameters,
)


# Create ServerApp
app = ServerApp(
    config=ServerConfig(num_rounds=3),
    strategy=strategy,
)
