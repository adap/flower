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

from typing import Dict, List, Optional, Tuple, Union

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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class DPWrapper_fixed_clipping(Strategy):
    """Wrapper for Configuring a Strategy for Central DP with Fixed Clipping.

    Parameters
    ----------
    strategy: Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier: float, optional
        The noise multiplier for the Gaussian mechanism for model updates.
        A value of 1.0 or higher is recommended for strong privacy.
    clip_norm: float
        The value of the clipping norm.
    num_sampled_clients: int
        The number of clients that are sampled on each round.
    """

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clip_norm: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__()

        self.strategy = strategy

        if noise_multiplier < 0:
            raise Exception("The noise multiplier should be a non-negative value.")

        if clip_norm <= 0:
            raise Exception("The clipping threshold should be a positive value.")

        if num_sampled_clients <= 0:
            raise Exception("The clipping threshold should be a positive value.")

        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        self.num_sampled_clients = num_sampled_clients

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"DPWrapper_fixed_clipping(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters using given strategy."""
        return self.strategy.initialize_parameters(client_manager)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        return self.strategy.configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        return self.strategy.configure_evaluate(
            server_round, parameters, client_manager
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results using unweighted aggregation."""
        if failures:
            return None, {}

        # Extract model updates
        all_updates = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        # Clip updates
        clipped_updates = self._clip_model_updates(all_updates)

        # Update the results with clipped updates
        updated_results = [
            (
                client,
                FitRes(
                    fit_res.status,
                    ndarrays_to_parameters(clipped_update),
                    fit_res.num_examples,
                    fit_res.metrics,
                ),
            )
            for (client, fit_res), clipped_update in zip(results, clipped_updates)
        ]

        # Pass the clipped updates for aggregation
        aggregated_updates, metrics = self.strategy.aggregate_fit(
            server_round, updated_results, failures
        )

        # Add Gaussian noise to the aggregated parameters
        if aggregated_updates:
            aggregated_updates = self._add_noise_to_updates(aggregated_updates)

        return aggregated_updates, metrics

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

    def _clip_model_updates(self, updates: NDArrays) -> List[Parameters]:
        """Clip model parameters based on the computed clip_norm."""
        clip_norm = self._get_update_norm(updates)
        clipped_updates = []

        for update in updates:
            clipped_update = {
                key: np.clip(value, -clip_norm, clip_norm)
                for key, value in update.items()
            }
            clipped_updates.append(clipped_update)

        return clipped_updates

    def _get_update_norm(update: NDArrays) -> float:
        flattened_update = np.concatenate(
            [np.asarray(sub_update).flatten() for sub_update in update]
        )
        return float(np.linalg.norm(flattened_update))

    def _add_noise_to_updates(self, parameters: Parameters) -> Parameters:
        """Add Gaussian noise to model updates."""
        return ndarrays_to_parameters(
            self.add_gaussian_noise(
                parameters_to_ndarrays(parameters),
                self.noise_multiplier * self.clip_norm,
            )
        )

    def _add_gaussian_noise(update: NDArrays, std_dev: float) -> NDArrays:
        update_noised = [
            layer + np.random.normal(0, std_dev, layer.shape) for layer in update
        ]
        return update_noised
