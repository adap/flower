"""Strategy class for FedNova."""

from functools import reduce
from logging import WARNING

import numpy as np
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class FedNovaStrategy(FedAvg):
    """Custom FedAvg strategy with fednova based configuration and aggregation."""

    def aggregate_fit(
        self,
        server_round: int,
        server_params: NDArrays,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        total_samples = sum([fit_res.num_examples for _, fit_res in results])
        c = sum(
            [
                fit_res.metrics["a_i"] * fit_res.num_examples / total_samples
                for _, fit_res in results
            ]
        )
        new_weights_results = [
            (result[0], c * (fit_res.num_examples / total_samples))
            for result, (_, fit_res) in zip(weights_results, results)
        ]

        # Aggregate grad updates, t_eff*(sum_i(p_i*\eta*d_i))
        grad_updates_aggregated = aggregate(new_weights_results)
        # Final parameters = server_params - grad_updates_aggregated
        parameters_aggregated = [
            server_param - grad_update
            for server_param, grad_update in zip(server_params, grad_updates_aggregated)
        ]

        parameters_aggregated = ndarrays_to_parameters(parameters_aggregated)
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Implement custom aggregate function for FedNova."""
    # Create a list of weights, each multiplied by the weight_factor
    weighted_weights = [
        [layer * factor for layer in weights] for weights, factor in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime
