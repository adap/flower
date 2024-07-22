"""fedvae: A Flower app for Federated Variational Autoencoder."""

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    # Construct ServerConfig
    num_rounds = int(context.run_config["num_server_rounds"])
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
