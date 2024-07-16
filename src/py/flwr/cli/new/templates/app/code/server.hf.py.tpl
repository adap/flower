"""$project_name: A Flower / HuggingFace Transformers app."""

from flwr.common import Context
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


def server_fn(context: Context):

    # Read from config
    num_rounds = int(context.run_config["num-server-rounds"])

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
