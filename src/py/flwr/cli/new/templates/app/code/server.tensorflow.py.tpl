"""$project_name: A Flower / $framework_str app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from $import_name.task import load_model


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    # Get parameters to initialize global model
    parameters = ndarrays_to_parameters(load_model().get_weights())

    # Define strategy
    strategy = strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
