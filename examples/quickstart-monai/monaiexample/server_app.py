from typing import List, Tuple

from flwr.common import Metrics, Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context):

    # Define strategy
    strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

    config = ServerConfig(num_rounds=3)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
