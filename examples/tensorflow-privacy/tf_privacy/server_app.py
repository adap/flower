"""tf_privacy: Training with Sample-Level Differential Privacy using TensorFlow-Privacy Engine."""

from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .task import load_model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    parameters = ndarrays_to_parameters(load_model().get_weights())
    strategy = FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average,
        initial_parameters=parameters,
    )
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
