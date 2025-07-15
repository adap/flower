"""Optionally define a custom strategy.

Needed only when the strategy is not yet implemented in Flower or because you want to
extend or modify the functionality of an existing strategy.
"""

import gc
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import torch
from flwr.common import (
    FitRes,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate


class CustomFedAvg(fl.server.strategy.FedAvg):
    """Custom strategy to aggregate using metrics instead of number of samples."""

    def __init__(self, *args, weight_strategy, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.weight_strategy = weight_strategy

    def aggregate_fit(
        self,
        _: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Union[bool, bytes, float, int, str]]]:
        """Aggregate results using different weighing metrics (train_loss or WER)."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        key_name = "train_loss" if self.weight_strategy == "loss" else "wer"
        weights = None

        # Define ratio merge
        if self.weight_strategy == "num":
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            weights = aggregate(weights_results)
        else:
            weights_results = [
                (
                    parameters_to_ndarrays(fit_res.parameters),
                    int(fit_res.metrics[key_name]),
                )
                for _, fit_res in results
            ]
            weights = aggregate(weights_results)

        # Free memory for next round
        del results, weights_results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return ndarrays_to_parameters(weights), {}
