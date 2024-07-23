from typing import List, Tuple

from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.common import Context, Metrics
from flwr.server.strategy import FedAvg


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour.

    You can use settings in `context.run_config` to parameterize the
    construction of all elements (e.g the strategy or the number of rounds)
    wrapped in the returned ServerAppComponents object.
    """

    # Define the strategy
    # We pass a callback to process the metrics returned by a client's
    # evaluate() method. Similarly, we set the fraction of clients to
    # sample for federated evaluation at run time based on the value
    # defined in the pyproject.toml (or overrided when calling `flwr run`.)
    fraction_eval = context.run_config["fraction-evaluate"]
    strategy = FedAvg(
        fraction_evaluate=fraction_eval,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Construct ServerConfig
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
