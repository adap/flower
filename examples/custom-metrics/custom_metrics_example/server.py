"""custom_metrics_example: A Flower app for custom metrics."""

import numpy as np

from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg


# Define metrics aggregation function
def average_metrics(metrics):
    """Aggregate metrics from multiple clients by calculating mean averages.

    Parameters:
    - metrics (list): A list containing tuples, where each tuple represents metrics for a client.
                    Each tuple is structured as (num_examples, metric), where:
                    - num_examples (int): The number of examples used to compute the metrics.
                    - metric (dict): A dictionary containing custom metrics provided as `output_dict`
                                    in the `evaluate` method from `client.py`.

    Returns:
    A dictionary with the aggregated metrics, calculating mean averages. The keys of the
    dictionary represent different metrics, including:
    - 'accuracy': Mean accuracy calculated by TensorFlow.
    - 'acc': Mean accuracy from scikit-learn.
    - 'rec': Mean recall from scikit-learn.
    - 'prec': Mean precision from scikit-learn.
    - 'f1': Mean F1 score from scikit-learn.

    Note: If a weighted average is required, the `num_examples` parameter can be leveraged.

    Example:
        Example `metrics` list for two clients after the last round:
        [(10000, {'prec': 0.108, 'acc': 0.108, 'f1': 0.108, 'accuracy': 0.1080000028014183, 'rec': 0.108}),
        (10000, {'f1': 0.108, 'rec': 0.108, 'accuracy': 0.1080000028014183, 'prec': 0.108, 'acc': 0.108})]
    """

    # Here num_examples are not taken into account by using _
    accuracies_tf = np.mean([metric["accuracy"] for _, metric in metrics])
    accuracies = np.mean([metric["acc"] for _, metric in metrics])
    recalls = np.mean([metric["rec"] for _, metric in metrics])
    precisions = np.mean([metric["prec"] for _, metric in metrics])
    f1s = np.mean([metric["f1"] for _, metric in metrics])

    return {
        "accuracy": accuracies_tf,
        "acc": accuracies,
        "rec": recalls,
        "prec": precisions,
        "f1": f1s,
    }


# strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=average_metrics)
#
#
# # Start Flower server
# fl.server.start_server(
#     server_address="0.0.0.0:8080",
#     config=fl.server.ServerConfig(num_rounds=3),
#     strategy=strategy,
# )


def server_fn(context: Context) -> ServerAppComponents:
    # Define strategy and the custom aggregation function for the evaluation metrics
    strategy = FedAvg(evaluate_metrics_aggregation_fn=average_metrics)

    # Construct ServerConfig
    num_rounds = int(context.run_config["num_server_rounds"])
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
