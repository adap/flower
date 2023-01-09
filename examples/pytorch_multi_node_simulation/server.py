from typing import List, Tuple
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
from multi_node_strategy import MultiNodeWrapper
import flwr


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
aggregating_strategy = FedAvg()
multi_node_strategy = MultiNodeWrapper(
    num_virtual_clients_fit_per_round=0,
    num_virtual_clients_fit_total=0,
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
