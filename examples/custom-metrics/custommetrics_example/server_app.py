"""custommetrics_example: A Flower / TensorFlow app for custom metrics."""

import numpy as np
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from custommetrics_example.task import get_model, get_parameters


# Define metrics aggregation function
def average_metrics(metrics):
    # pylint: disable=C0301
    """Aggregate metrics from multiple clients by calculating mean averages.

    Parameters
    ----------
    metrics : list
        A list containing tuples, where each tuple represents metrics for a client.
        Each tuple is structured as (num_examples, metric), where:
        - num_examples (int) : The number of examples used to compute the metrics.
        - metric (dict) : A dictionary containing custom metrics provided as
                          `output_dict` in the `evaluate` method from `client.py`.

    Returns
    -------
    dict
        A dictionary with the aggregated metrics, calculating mean averages.
        The keys of the dictionary represent different metrics, including:
        - 'accuracy': Mean accuracy calculated by TensorFlow.
        - 'acc': Mean accuracy from scikit-learn.
        - 'rec': Mean recall from scikit-learn.
        - 'prec': Mean precision from scikit-learn.
        - 'f1': Mean F1 score from scikit-learn.

        Note: If a weighted average is required, the `num_examples` parameter can be
        leveraged.

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


def server_fn(context: Context) -> ServerAppComponents:
    """Construct components that set the ServerApp behaviour."""

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]

    model = get_model()
    ndarrays = get_parameters(model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy and the custom aggregation function for the evaluation metrics
    strategy = FedAvg(
        evaluate_metrics_aggregation_fn=average_metrics,
        initial_parameters=global_model_init,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
