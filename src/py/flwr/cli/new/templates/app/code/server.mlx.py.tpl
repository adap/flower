"""$project_name: A Flower / $framework_str app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from $import_name.task import MLP, get_params


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    num_classes = 10
    num_layers = context.run_config["num-layers"]
    input_dim = context.run_config["input-dim"]
    hidden_dim = context.run_config["hidden-dim"]

    # Initialize global model
    model = MLP(num_layers, input_dim, hidden_dim, num_classes)
    params = get_params(model)
    initial_parameters = ndarrays_to_parameters(params)

    # Define strategy
    strategy = FedAvg(initial_parameters=initial_parameters)
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
