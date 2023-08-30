import pickle
import torch
from typing import List, Callable
import numpy as np
import flwr.common
from flwr.common import (
    FitIns,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    NDArrays,
)
from flwr.server.strategy import Strategy
from models import Net


def aggregate(weights: List[NDArrays], s: float) -> NDArrays:
    """Computes average of the clients' weights."""
    averaged_weights = [
        np.sum(layer_updates, axis=0) / s for layer_updates in zip(*weights)
    ]
    return averaged_weights


def create_pattern(dim: int, cohort_size: int, sparsity: int):
    """Creates compression pattern."""

    q = torch.zeros(size=(dim, cohort_size))
    if dim >= cohort_size / sparsity:
        k = 0
        for i in range(dim):
            for j in range(sparsity):
                q[i, k] = 1
                k = (k + 1) % cohort_size
    else:
        k = 0
        for j in range(sparsity):
            for i in range(dim):
                q[i, k] = 1
                k += 1

    return q


def shuffle_columns(q: torch.Tensor):
    q = q[:, torch.randperm(q.size()[1])]
    return q


def save_client_mask_to_file(cid, mask):
    """Saves client mask to file, so client can read it later."""
    with open(f"{cid}_mask.bin", "wb") as f:
        pickle.dump(mask, f, protocol=pickle.HIGHEST_PROTOCOL)


class TamunaStrategy(Strategy):
    def __init__(
        self,
        clients_per_round: int,
        epochs_per_round: List[int],
        eta: float,
        s: int,
        evaluate_fn: Callable,
    ) -> None:
        self.clients_per_round = clients_per_round
        self.epochs_per_round = epochs_per_round
        self.evaluate_fn = evaluate_fn
        self.eta = eta
        self.s = s
        self.dim = None
        self.server_model = None
        self.compression_pattern = None

    def initialize_parameters(self, client_manager):
        """Initialize the server model."""
        self.server_model = Net()
        self.dim = sum(p.numel() for p in self.server_model.parameters())

        self.compression_pattern = create_pattern(
            self.dim, self.clients_per_round, self.s
        )  # dim x cohort_size

        ndarrays = [
            val.cpu().numpy() for _, val in self.server_model.state_dict().items()
        ]
        return flwr.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Sample clients and create compression pattern for each of them."""
        sampled_clients = client_manager.sample(self.clients_per_round)
        config = {"epochs": self.epochs_per_round[server_round - 1], "eta": self.eta}

        self.compression_pattern = shuffle_columns(self.compression_pattern)

        for i in range(self.clients_per_round):
            save_client_mask_to_file(sampled_clients[i].cid, self.compression_pattern[:, i])

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in sampled_clients]

    def aggregate_fit(self, server_round, results, failures):
        """Average the clients' weights."""
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights, self.s))
        return parameters_aggregated, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters):
        return self.evaluate_fn(parameters)
