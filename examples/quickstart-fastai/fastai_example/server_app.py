"""fastai_example: A Flower / Fastai app."""

from typing import List, Tuple

from flwr.common import Context, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


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
    # Construct ServerConfig
    num_rounds = context.run_config["num_server_rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    # Define strategy
    strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
