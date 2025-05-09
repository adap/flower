"""Flower Server."""
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('always')

from typing import List, Tuple

from flwr.common import Metrics

# Define metric aggregation function

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
     for num_examples, m in metrics: print("Client",m["cid"]," its recieved accuracy", m["accuracy"])
     accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
     examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
     return {"accuracy": sum(accuracies) / sum(examples)}



def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {

        "current_round": server_round ,

    }
    return config




