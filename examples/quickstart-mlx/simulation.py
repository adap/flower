import argparse

import flwr as fl
from flwr.common import Metrics
from datasets.utils.logging import disable_progress_bar

from client import get_client_fn


# Define metric aggregation function
def weighted_average(metrics) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


if __name__ == "__main__":
    disable_progress_bar()

    parser = argparse.ArgumentParser(
        "Perform FL simulation on a simple MLP on MNIST with MLX and Flower."
    )
    parser.add_argument("--gpu", action="store_true", help="Use the Metal back-end.")
    parser.add_argument(
        "--num_clients", type=int, default=2, help="Number of clients to train on."
    )
    parser.add_argument(
        "--num_rounds", type=int, default=50, help="Number of rounds to train for."
    )
    args = parser.parse_args()

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    )

    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(args.num_clients),
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )
