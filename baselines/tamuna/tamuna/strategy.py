from typing import List, Tuple, Union, Callable
import numpy as np
import flwr.common
from collections import OrderedDict
from flwr.common import Metrics
from flwr.common import (
    FitIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    NDArrays,
    NDArray,
)
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy, FedAvg
from tamuna.models import Net, test
from functools import reduce
import torch


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


def aggregate(weights: List[NDArrays]) -> NDArrays:
    """Computes average."""
    num_clients = len(weights)
    averaged_weights = [
        np.sum(layer_updates, axis=0) / num_clients for layer_updates in zip(*weights)
    ]
    return averaged_weights


class TamunaStrategy(Strategy):
    def __init__(
        self, clients_per_round: int, epochs_per_round: List[int], evaluate_fn: Callable
    ) -> None:
        self.clients_per_round = clients_per_round
        self.epochs_per_round = epochs_per_round
        self.evaluate_fn = evaluate_fn
        self.server_model = None

    def initialize_parameters(self, client_manager):
        self.server_model = Net()
        ndarrays = [
            val.cpu().numpy() for _, val in self.server_model.state_dict().items()
        ]
        return flwr.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(self, server_round: int, parameters, client_manager):
        clients = client_manager.sample(self.clients_per_round)
        config = {"epochs": self.epochs_per_round[server_round - 1]}
        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(self, server_round, results, failures):
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights))
        return parameters_aggregated, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters):
        if server_round == 0:
            return None
        return self.evaluate_fn(parameters)
