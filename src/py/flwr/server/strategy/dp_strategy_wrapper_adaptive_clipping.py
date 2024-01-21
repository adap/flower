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
        clipped_count_stddev: Optional[float] = None
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
            raise Exception("The target clipped quantile must be between 0 and 1 (inclusive).")

        if clip_norm_lr <= 0:
            raise Exception("The learning rate must be positive.")

        if clipped_count_stddev is None:
            if clipped_count_stddev < 0:
                raise Exception("The `clipped_count_stddev` must be non-negative.")

        strategy = strategy,
        noise_multiplier=noise_multiplier,
        num_sampled_clients=num_sampled_clients,
        initial_clip_norm=initial_clip_norm,
        target_clipped_quantile=target_clipped_quantile,
        clip_norm_lr=clip_norm_lr,
        clipped_count_stddev=clipped_count_stddev,






