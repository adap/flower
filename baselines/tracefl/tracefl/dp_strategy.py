"""Custom Differential Privacy Strategy for TraceFL."""

from logging import INFO, WARNING
from typing import Optional, Union

import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.differential_privacy import add_gaussian_noise_to_params, compute_stdv
from flwr.common.differential_privacy_constants import (
    CLIENTS_DISCREPANCY_WARNING,
    KEY_CLIPPING_NORM,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy
from tracefl.dp_utils import safe_clip_inputs_inplace


class TraceFLDifferentialPrivacy(Strategy):
    """Custom differential privacy strategy for TraceFL.

    This strategy uses our safe implementation of clipping that handles zero norm cases.
    """

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__()

        self.strategy = strategy

        # Validate DP parameters
        if noise_multiplier <= 0:
            raise ValueError(
                "The noise multiplier must be positive when using DP. "
                "Set to -1 to disable DP completely."
            )

        if clipping_norm <= 0:
            raise ValueError(
                "The clipping norm must be positive when using DP. "
                "Set to -1 to disable DP completely."
            )

        if num_sampled_clients <= 0:
            raise ValueError(
                "The number of sampled clients should be a positive value."
            )

        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients
        self.current_round_params: NDArrays = []

        # Log DP configuration
        log(
            INFO,
            "Initialized DP strategy with "
            "noise_multiplier=%.4f, clipping_norm=%.4f, num_clients=%d",
            self.noise_multiplier,
            self.clipping_norm,
            self.num_sampled_clients,
        )

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "TraceFL Differential Privacy Strategy Wrapper"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters using given strategy."""
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        additional_config = {KEY_CLIPPING_NORM: self.clipping_norm}
        inner_strategy_config_result = self.strategy.configure_fit(
            server_round, parameters, client_manager
        )
        for _, fit_ins in inner_strategy_config_result:
            fit_ins.config.update(additional_config)
        return inner_strategy_config_result

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Compute the updates, clip, and pass them for aggregation.

        Afterward, add noise to the aggregated parameters.
        """
        if failures:
            return None, {}

        if len(results) != self.num_sampled_clients:
            log(
                WARNING,
                CLIENTS_DISCREPANCY_WARNING,
                len(results),
                self.num_sampled_clients,
            )

        for _, res in results:
            param = parameters_to_ndarrays(res.parameters)
            # Compute update
            model_update = [
                np.subtract(x, y) for (x, y) in zip(param, self.current_round_params)
            ]

            # Use our safe clipping implementation
            safe_clip_inputs_inplace(model_update, self.clipping_norm)

            log(
                INFO,
                "aggregate_fit: parameters are clipped by value: %.4f.",
                self.clipping_norm,
            )

            # Add clipped update back to original parameters
            for i, _ in enumerate(self.current_round_params):
                param[i] = self.current_round_params[i] + model_update[i]

            # Convert back to parameters
            res.parameters = ndarrays_to_parameters(param)

        # Pass the new parameters for aggregation
        aggregated_params, metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )

        # Add Gaussian noise to the aggregated parameters
        if aggregated_params:
            aggregated_params = add_gaussian_noise_to_params(
                aggregated_params,
                self.noise_multiplier,
                self.clipping_norm,
                self.num_sampled_clients,
            )

            log(
                INFO,
                "aggregate_fit: central DP noise with %.4f stdev added",
                compute_stdv(
                    self.noise_multiplier, self.clipping_norm, self.num_sampled_clients
                ),
            )

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using the given strategy."""
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function from the strategy."""
        return self.strategy.evaluate(server_round, parameters)
