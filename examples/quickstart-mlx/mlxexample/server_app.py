"""mlxexample: A Flower / MLX app."""

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context
from flwr.server.strategy import FedAvg


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Define strategy
    strategy = FedAvg()

    # Construct ServerConfig
    num_rounds = context.run_config["num_server_rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
