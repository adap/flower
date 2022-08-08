"""DP-FedAvg [McMahan et al., 2018] strategy.

Paper: https://arxiv.org/pdf/1710.06963.pdf
"""


from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Scalar
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class DPFixedClipStrategy(Strategy):
    """Wrapper for configuring a Strategy for DP with Fixed Clipping."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        num_sampled_clients: int,
        clip_norm: float,
        noise_multiplier: float = 1,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        # Doing fixed-size subsampling as in https://arxiv.org/abs/1905.03871.
        self.num_sampled_clients = num_sampled_clients
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier

    def __repr__(self) -> str:
        rep = "Strategy with DP with Fixed Clipping enabled."
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        client_instructions = self.strategy.configure_fit(
            rnd, parameters, client_manager
        )

        for _, fit_ins in client_instructions:
            fit_ins.config["clip_norm"] = self.clip_norm

        return client_instructions

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(rnd, parameters, client_manager)

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if failures:
            return None, {}
        # Forcing unweighted aggregation, as in https://arxiv.org/abs/1905.03871.
        for _, fit_res in results:
            fit_res.num_examples = 1

        aggregate_params, aggregate_metrics = self.strategy.aggregate_fit(
            rnd, results, failures
        )

        if aggregate_params is None:
            return None, {}

        aggregate_weights = parameters_to_ndarrays(aggregate_params)
        noised_weights = [
            layer
            + np.random.normal(
                0,
                self.noise_multiplier * self.clip_norm / self.num_sampled_clients,
                layer.shape,
            )
            for layer in aggregate_weights
        ]
        return ndarrays_to_parameters(noised_weights), aggregate_metrics

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return self.strategy.aggregate_evaluate(rnd, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return self.strategy.evaluate(server_round, parameters)
