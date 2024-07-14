"""$project_name: A Flower / NumPy app."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def server_fn(context: Context):
    # Define strategy
    strategy = FedAvg()
    config = ServerConfig(num_rounds=3)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
