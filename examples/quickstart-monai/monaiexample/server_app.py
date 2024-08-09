"""monaiexample: A Flower / MONAI app."""

from typing import List, Tuple

from flwr.common import Metrics, Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from monaiexample.task import load_model, get_params


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):

    # Init model
    model = load_model()

    # Convert model parameters to flwr.common.Parameters
    ndarrays = get_params(model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy
    fraction_fit = context.run_config["fraction-fit"]
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=global_model_init,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
