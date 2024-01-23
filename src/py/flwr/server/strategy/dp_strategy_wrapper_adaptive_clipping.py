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
"""Central DP with client side adaptive clipping.

Paper (Andrew et al.): https://arxiv.org/pdf/1905.03871.pdf
"""

import math
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from flwr.common import (
    EvaluateIns,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy


class DPStrategyWrapperClientSideAdaptiveClipping(Strategy):
    """Wrapper for Configuring a Strategy for Central DP with Adaptive Clipping.

    The clipping is at the client side.

    Parameters
    ----------
    strategy: Strategy
        The strategy to which DP functionalities will be added by this wrapper.
    noise_multiplier: float
        The noise multiplier for the Gaussian mechanism for model updates.
    num_sampled_clients: int
        The number of clients that are sampled on each round.
    initial_clip_norm: float
        The initial value of clipping norm. Deafults to 0.1.
        Andrew et al. recommends to set to 0.1.
    target_clipped_quantile: float
        The desired quantile of updates which should be clipped. Defaults to 0.5.
    clip_norm_lr: float
        The learning rate for the clipping norm adaptation. Defaults to 0.2.
        Andrew et al. recommends to set to 0.2.
    clipped_count_stddev: float
        The stddev of the noise added to the count of updates currently below the estimate.
        Andrew et al. recommends to set to `expected_num_records/20`
    use_geometric_update: bool
        Use geometric updating of clip. Defaults to True.
        It is recommended by Andrew et al. to use it.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        num_sampled_clients: int,
        initial_clip_norm: float = 0.1,
        target_clipped_quantile: float = 0.5,
        clip_norm_lr: float = 0.2,
        clipped_count_stddev: Optional[float] = None,
    ) -> None:
        super().__init__()

        if strategy is None:
            raise Exception("The passed strategy is None.")

        if noise_multiplier < 0:
            raise Exception("The noise multiplier should be a non-negative value.")

        if num_sampled_clients <= 0:
            raise Exception("The number of sampled clients should be a positive value.")

        if initial_clip_norm <= 0:
            raise Exception("The initial clip norm should be a positive value.")

        if not 0 <= target_clipped_quantile <= 1:
            raise Exception(
                "The target clipped quantile must be between 0 and 1 (inclusive)."
            )

        if clip_norm_lr <= 0:
            raise Exception("The learning rate must be positive.")

        if clipped_count_stddev is not None:
            if clipped_count_stddev < 0:
                raise Exception("The `clipped_count_stddev` must be non-negative.")

        self.strategy = strategy
        self.num_sampled_clients = num_sampled_clients
        self.clip_norm = initial_clip_norm
        self.target_clipped_quantile = target_clipped_quantile
        self.clip_norm_lr = clip_norm_lr
        self.clipped_count_stddev, self.noise_multiplier = self._compute_noise_params(
            clipped_count_stddev
        )

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = "DP Strategy Wrapper with Client Side Adaptive Clipping"
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
        """Aggregate training results and update clip norms."""
        if failures:
            return None, {}

        aggregated_params, metrics = self.strategy.aggregate_fit(
            server_round, results, failures
        )
        self._update_clip_norm(results)
        if aggregated_params:
            aggregated_params = self._add_noise_to_updates(aggregated_params)
        return aggregated_params

    def _update_clip_norm(self, results: List[Tuple[ClientProxy, FitRes]]) -> None:
        # calculate the number of clients which set the norm indicator bit
        norm_bit_set_count = 0
        for client_proxy, fit_res in results:
            if "dpfedavg_norm_bit" not in fit_res.metrics:
                raise Exception(
                    f"Indicator bit not returned by client with id {client_proxy.cid}."
                )
            if fit_res.metrics["dpfedavg_norm_bit"]:
                norm_bit_set_count += 1
        # Noising the count
        noised_norm_bit_set_count = float(
            np.random.normal(norm_bit_set_count, self.clipped_count_stddev)
        )

        noised_norm_bit_set_fraction = noised_norm_bit_set_count / len(results)
        # Geometric update
        self.clip_norm *= math.exp(
            -self.clip_norm_lr
            * (noised_norm_bit_set_fraction - self.target_clipped_quantile)
        )

    def _add_noise_to_updates(self, parameters: Parameters) -> Parameters:
        """Add Gaussian noise to model params."""
        return ndarrays_to_parameters(
            self.add_gaussian_noise(
                parameters_to_ndarrays(parameters),
                float(
                    (self.noise_multiplier * self.clip_norm)
                    / self.num_sampled_clients ** (0.5)
                ),
            )
        )

    @staticmethod
    def _compute_noise_params(
        noise_multiplier: float,
        num_sampled_clients: float,
        clipped_count_stddev: Optional[float],
    ):
        """Compute noising parameters for the adaptive clipping.

        paper: https://arxiv.org/abs/1905.03871
        """
        if noise_multiplier > 0:
            if clipped_count_stddev is None:
                clipped_count_stddev = num_sampled_clients / 20
            if noise_multiplier >= 2 * clipped_count_stddev:
                raise ValueError(
                    f"If not specified, `clipped_count_stddev` is set to `num_sampled_clients`/20 by default. "
                    f"This value ({num_sampled_clients / 20}) is too low to achieve the desired effective `noise_multiplier` ({noise_multiplier})."
                    f"Consider increasing `clipped_count_stddev` or decreasing `noise_multiplier`."
                )
            noise_multiplier_value = (
                noise_multiplier ** (-2) - (2 * clipped_count_stddev) ** (-2)
            ) ** -0.5

            adding_noise = noise_multiplier_value / noise_multiplier
            if adding_noise >= 2:
                warnings.warn(
                    f"A significant amount of noise ({adding_noise}) has to be added. Consider increasing"
                    f" `clipped_count_stddev` or `num_sampled_clients`.",
                    stacklevel=2,
                )

        else:
            if clipped_count_stddev is None:
                clipped_count_stddev = 0.0
            noise_multiplier_value = 0.0

        return clipped_count_stddev, noise_multiplier_value
