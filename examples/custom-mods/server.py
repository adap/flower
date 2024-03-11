from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    train_losses = [
        num_examples * float(m["train_loss"]) for num_examples, m in metrics
    ]
    train_accuracies = [
        num_examples * float(m["train_accuracy"]) for num_examples, m in metrics
    ]
    val_losses = [num_examples * float(m["val_loss"]) for num_examples, m in metrics]
    val_accuracies = [
        num_examples * float(m["val_accuracy"]) for num_examples, m in metrics
    ]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Select all available clients
    fraction_evaluate=0.0,  # Disable evaluation
    min_available_clients=2,
    fit_metrics_aggregation_fn=weighted_average,
)


# Run via `flower-server-app server:app`
app = fl.server.ServerApp(
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)
