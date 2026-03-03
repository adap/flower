"""fedhomo: A Flower Baseline."""

import logging
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from fedhomo.model import get_model, get_weights
from fedhomo.strategy import HomomorphicFedAvg

log = logging.getLogger(__name__)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Compute weighted average of accuracy across clients."""
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def server_fn(context: Context) -> ServerAppComponents:
    """Construct server components given the context."""
    num_rounds = int(context.run_config["num-server-rounds"])
    fraction_fit = float(context.run_config["fraction-fit"])
    dataset = context.run_config["dataset"]

    parameters = ndarrays_to_parameters(get_weights(get_model(dataset)))

    strategy = HomomorphicFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        accept_failures=True,
    )

    log.info("Server: %d rounds, fraction_fit=%.2f, dataset=%s", num_rounds, fraction_fit, dataset)

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=num_rounds),
    )


app = ServerApp(server_fn=server_fn)
