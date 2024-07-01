"""$project_name: A Flower / TensorFlow app."""

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg

from $import_name.task import load_model

# Define config
config = ServerConfig(num_rounds=1)

parameters = ndarrays_to_parameters(load_model().get_weights())

# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,
    initial_parameters=parameters,
)


# Create ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)
