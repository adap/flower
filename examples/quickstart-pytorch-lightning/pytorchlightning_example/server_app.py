"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    # Construct ServerConfig
    num_rounds = context.run_config["num_server_rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    # Define strategy
    strategy = FedAvg(fraction_fit=0.5, fraction_evaluate=0.5)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
