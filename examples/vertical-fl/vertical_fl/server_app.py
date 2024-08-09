from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from vertical_fl.strategy import Strategy
from vertical_fl.task import get_partitions_and_label

_, label = get_partitions_and_label()


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""
    # Define the strategy
    strategy = Strategy(label)

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Start Flower server
app = ServerApp(server_fn=server_fn)
