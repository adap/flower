import os
import pickle
from logging import WARNING
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from flwr.common import (
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.common.typing import FitRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from omegaconf import DictConfig


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


class FedDyn(FedAvg):
    """Applying dynamic regularization in FedDyn paper."""

    def __init__(self, cfg: DictConfig, net: nn.Module, *args, **kwargs):
        self.cfg = cfg
        self.h = [np.zeros(v.shape) for (k, v) in net.state_dict().items()]
        self.prev_grads = [
            {k: torch.zeros(v.numel()) for (k, v) in net.named_parameters()}
        ] * cfg.num_clients

        if not os.path.exists("prev_grads"):
            os.makedirs("prev_grads")

        for idx in range(cfg.num_clients):
            with open(f"prev_grads/client_{idx}", "wb") as f:
                pickle.dump(self.prev_grads[idx], f)

        self.is_weight = []

        # tagging real weights / biases
        for k in net.state_dict().keys():
            if "weight" not in k and "bias" not in k:
                self.is_weight.append(False)
            else:
                self.is_weight.append(True)

        super().__init__(*args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        origin: NDArrays,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        for idx in range(self.cfg.num_clients):
            with open(f"prev_grads/client_{idx}", "rb") as f:
                self.prev_grads[idx] = pickle.load(f)

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        parameters_aggregated = ndarrays_to_parameters(
            aggregate(weights_results, origin, self.h, self.is_weight, self.cfg)
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated


def aggregate(
    results: List[Tuple[NDArrays, int]],
    origin: NDArrays,
    h: List,
    is_weight: List,
    cfg: DictConfig,
) -> NDArrays:
    param_count = [0] * len(origin)
    weights_sum = [np.zeros(v.shape) for v in origin]

    # summation & counting of parameters
    for weight, _ in results:
        for i, layer in enumerate(weight):
            weights_sum[i] += layer
            param_count[i] += 1

    # update parameters
    for i, weight in enumerate(weights_sum):
        if param_count[i] > 0:
            weight = weight / param_count[i]
            # print(np.isscalar(weight))

            # update h variable for FedDyn
            h[i] = (
                h[i]
                - cfg.fit_config.alpha
                * param_count[i]
                * (weight - origin[i])
                / cfg.num_clients
            )

            # applying h only for weights / biases
            if is_weight[i] and cfg.fit_config.feddyn:
                weights_sum[i] = weight - h[i] / cfg.fit_config.alpha
            else:
                weights_sum[i] = weight

        else:
            weights_sum[i] = origin[i]

    return weights_sum
