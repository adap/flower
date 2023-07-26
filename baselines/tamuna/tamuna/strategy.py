from typing import List, Tuple, Union

from flwr.common import Metrics
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    print("here and nothing is breaking!!!")
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
