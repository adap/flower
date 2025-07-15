"""fastai_example: A Flower / Fastai app."""

from typing import List, Tuple

from fastai.vision.all import squeezenet1_1
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fastai_example.task import get_params


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average metric values."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""

    # Let's define the global model and pass it to the strategy
    # Note this is optional.
    parameters = ndarrays_to_parameters(get_params(squeezenet1_1()))

    # Define strategy
    fraction_fit = context.run_config["fraction-fit"]
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=0.5,
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
