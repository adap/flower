from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
from pathlib import Path


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=weighted_average)

# Start Flower server
hist = fl.server.start_driver(
    server_address="127.0.0.1:9091",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    root_certificates=Path("../client-auth/certificates/ca.crt").read_bytes(),
)

assert (hist.losses_distributed[0][1] / hist.losses_distributed[-1][1]) > 0.98
