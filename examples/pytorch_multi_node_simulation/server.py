from argparse import ArgumentParser
from typing import Dict, List, Tuple
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from flwr.common.typing import Scalar
from multi_node_strategy import MultiNodeWrapper
import flwr


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config: Dict[str, Scalar] = {
        "batch_size": 20,
        "current_round": server_round,
        "epochs": 1,
        "lr": 0.05,
        "weight_decay": 5e-4,
        "momentum": 0.9,
    }
    return config


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Flower Server adapted for Multi-Node Worker Setup.",
        description="This is a Flwr Server that connects to multiple workers each running virtual clients in a single process.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        required=True,
        help="Number of workers required for training.",
    )
    args = parser.parse_args()

    # Define strategy
    aggregating_strategy = FedAvg(
        fraction_evaluate=0.0,
        fraction_fit=1.0,
        min_available_clients=args.num_workers,
        min_fit_clients=args.num_workers,
        on_fit_config_fn=fit_config,
    )

    multi_node_strategy = MultiNodeWrapper(
        num_virtual_clients_fit_per_round=100,
        num_virtual_clients_fit_total=10965,
        num_virtual_clients_eval_per_round=0,
        num_virtual_clients_eval_total=0,
        aggregating_strategy=aggregating_strategy,
    )

    # Start Flower server
    flwr.server.start_server(
        server_address="0.0.0.0:8080",
        config=flwr.server.ServerConfig(num_rounds=5),
        strategy=multi_node_strategy,
    )
