"""basic-app: A Flower / NumPy app."""

from basic_app.task import get_dummy_model

from flwr.app import Context
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.serverapp import ServerApp


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Initial model
    model = get_dummy_model()
    dummy_parameters = ndarrays_to_parameters([model])

    # Define strategy
    strategy = FedAvg(initial_parameters=dummy_parameters)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
