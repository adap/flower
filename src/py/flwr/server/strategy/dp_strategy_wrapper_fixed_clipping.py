# Copyright 2020 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Central differential privacy with fixed clipping.

Papers: https://arxiv.org/pdf/1712.07557.pdf, https://arxiv.org/pdf/1710.06963.pdf
"""
import warnings
from typing import Dict, List, Optional, Tuple, Union

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
from flwr.common.differential_privacy import (
    add_gaussian_noise_to_params,
    compute_clip_model_update,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class DPStrategyWrapperServerSideFixedClipping(Strategy):
    """Wrapper for Configuring a Strategy for Central DP with Server Side Fixed
    Clipping.

    Parameters
    ----------
    strategy: Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier: float
        The noise multiplier for the Gaussian mechanism for model updates.
        A value of 1.0 or higher is recommended for strong privacy.
    clipping_norm: float
        The value of the clipping norm.
    num_sampled_clients: int
        The number of clients that are sampled on each round.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__()

        self.strategy = strategy

        if noise_multiplier < 0:
            raise ValueError("The noise multiplier should be a non-negative value.")

        if clipping_norm <= 0:
            raise ValueError("The clipping norm should be a positive value.")

        if num_sampled_clients <= 0:
            raise ValueError(
                "The number of sampled clients should be a positive value."
            )

        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self.num_sampled_clients = num_sampled_clients

        self.current_round_params: NDArrays = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "DP Strategy Wrapper with Server Side Fixed Clipping"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters using given strategy."""
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.current_round_params = parameters_to_ndarrays(parameters)
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Compute the updates, clip them, and pass them to the child strategy for
        aggregation.

        Afterward, add noise to the aggregated parameters.
        """
        if failures:
            return None, {}

        if len(results) != self.num_sampled_clients:
            warnings.warn(
                f"The number of clients returning parameters ({len(results)}) differs from "
                f"the number of sampled clients ({self.num_sampled_clients}). This could impact "
                f"the differential privacy guarantees of the system, potentially leading to privacy leakage "
                f"or inadequate noise calibration.",
                stacklevel=2,
            )
        for _, res in results:
            param = parameters_to_ndarrays(res.parameters)
            # Compute and clip update
            compute_clip_model_update(
                param, self.current_round_params, self.clipping_norm
            )
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

        return aggregated_params, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using the given strategy."""
        return self.strategy.aggregate_evaluate(server_round, results, failures)

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function from the strategy."""
        return self.strategy.evaluate(server_round, parameters)
