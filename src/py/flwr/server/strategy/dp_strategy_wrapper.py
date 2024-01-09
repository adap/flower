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
"""Central DP.

Papers: https://arxiv.org/pdf/1712.07557.pdf, https://arxiv.org/pdf/1710.06963.pdf
Note: unlike the above papers, we moved the clipping part to the server side.
"""
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


class DPStrategyWrapperFixedClipping(Strategy):
    """Wrapper for Configuring a Strategy for Central DP with Fixed Clipping.

    Parameters
    ----------
    strategy: Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier: float
        The noise multiplier for the Gaussian mechanism for model updates.
        A value of 1.0 or higher is recommended for strong privacy.
    clipping_threshold: float
        The value of the clipping threshold.
    num_sampled_clients: int
        The number of clients that are sampled on each round.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_threshold: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__()

        self.strategy = strategy

        if noise_multiplier < 0:
            raise Exception("The noise multiplier should be a non-negative value.")

        if clipping_threshold <= 0:
            raise Exception("The clipping threshold should be a positive value.")

        if num_sampled_clients <= 0:
            raise Exception("The clipping threshold should be a positive value.")

        self.noise_multiplier = noise_multiplier
        self.clipping_threshold = clipping_threshold
        self.num_sampled_clients = num_sampled_clients

        self.current_round_params: NDArrays = []

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "DP Strategy Wrapper with Fixed Clipping"
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
        """Aggregate training results using unweighted aggregation."""
        if failures:
            return None, {}

        # Extract all clients' model params
        clients_params = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]

        # Compute the updates
        all_clients_updates = self._compute_model_updates(clients_params)

        # Clip updates
        for client_update in all_clients_updates:
            client_update = self._clip_model_update(client_update)

        # Compute the new parameters with the clipped updates
        for client_param, client_update in zip(clients_params, all_clients_updates):
            self._update_clients_params(client_param, client_update)

        # Update the results with the new params
        updated_results = [
            (
                client,
                FitRes(
                    fit_res.status,
                    ndarrays_to_parameters(client_param),
                    fit_res.num_examples,
                    fit_res.metrics,
                ),
            )
            for (client, fit_res), client_param in zip(results, clients_params)
        ]

        # Pass the new parameters for aggregation
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

    def _clip_model_update(self, update: NDArrays) -> NDArrays:
        """Clip model update based on the computed clipping_threshold.

        FlatClip method of the paper: https://arxiv.org/pdf/1710.06963.pdf
        """
        update_norm = self._get_update_norm(update)
        scaling_factor = min(1, self.clipping_threshold / update_norm)
        update_clipped: NDArrays = [layer * scaling_factor for layer in update]
        return update_clipped

    @staticmethod
    def _get_update_norm(update: NDArrays) -> float:
        flattened_update = np.concatenate(
            [np.asarray(sub_update).flatten() for sub_update in update]
        )
        return float(np.linalg.norm(flattened_update))

    def _add_noise_to_updates(self, parameters: Parameters) -> Parameters:
        """Add Gaussian noise to model params."""
        return ndarrays_to_parameters(
            self._add_gaussian_noise(
                parameters_to_ndarrays(parameters),
                float((self.noise_multiplier * self.clipping_threshold)
                / self.num_sampled_clients ** (0.5)),
            )
        )

    @staticmethod
    def _add_gaussian_noise(update: NDArrays, std_dev: float) -> NDArrays:
        update_noised = [
            layer + np.random.normal(0, std_dev, layer.shape) for layer in update
        ]
        return update_noised

    def _compute_model_updates(
        self, all_clients_params: List[NDArrays]
    ) -> List[NDArrays]:
        all_client_updates = []
        for client_param in all_clients_params:
            client_update = [
                np.subtract(x, y)
                for (x, y) in zip(client_param, self.current_round_params)
            ]
            all_client_updates.append(client_update)
        return all_client_updates

    def _update_clients_params(
        self, client_param: NDArrays, client_update: NDArrays
    ) -> None:
        for i, _ in enumerate(self.current_round_params):
            client_param[i] = self.current_round_params[i] + client_update[i]
