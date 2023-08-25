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
from tamuna.models import Net


def aggregate(weights: List[NDArrays]) -> NDArrays:
    """Computes average."""
    num_clients = len(weights)
    averaged_weights = [
        np.sum(layer_updates, axis=0) / num_clients for layer_updates in zip(*weights)
    ]
    return averaged_weights


def create_pattern(dim: int, clients_per_round: int, s: int):
    return torch.ones(size=(dim, clients_per_round))


def save_client_mask_to_file(cid, mask):
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
        self.server_model = None

    def initialize_parameters(self, client_manager):
        self.server_model = Net()
        ndarrays = [
            val.cpu().numpy() for _, val in self.server_model.state_dict().items()
        ]
        return flwr.common.ndarrays_to_parameters(ndarrays)

    def configure_fit(self, server_round: int, parameters, client_manager):
        sampled_clients = client_manager.sample(self.clients_per_round)
        config = {"epochs": self.epochs_per_round[server_round - 1], "eta": self.eta}

        dim = sum(p.numel() for p in self.server_model.parameters())

        compression_pattern = create_pattern(
            dim, self.clients_per_round, self.s
        )  # dim x cohort_size

        for i in range(self.clients_per_round):
            save_client_mask_to_file(sampled_clients[i].cid, compression_pattern[:, i])

        fit_ins = FitIns(parameters, config)
        return [(client, fit_ins) for client in sampled_clients]

    def aggregate_fit(self, server_round, results, failures):
        weights = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights))
        return parameters_aggregated, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None, {}

    def evaluate(self, server_round: int, parameters):
        return self.evaluate_fn(parameters)
