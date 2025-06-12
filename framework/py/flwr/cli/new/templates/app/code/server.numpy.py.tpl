"""$project_name: A Flower / $framework_str app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from $import_name.task import get_dummy_model


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
