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
"""DPStrategyWrapperClientSideAdaptiveClipping tests."""


from flwr.common import FitRes, Parameters
from flwr.server.client_proxy import ClientProxy

from .dp_strategy_wrapper_client_side_adaptive_clipping import (
    DPStrategyWrapperClientSideAdaptiveClipping,
)
from .fedavg import FedAvg


def test_update_clip_norm() -> None:
    """Test _update_clip_norm method."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPStrategyWrapperClientSideAdaptiveClipping(strategy, 1.5, 5)

    results = [
        (ClientProxy("client1"), FitRes(Parameters(), {"dpfedavg_norm_bit": True})),
        (ClientProxy("client2"), FitRes(Parameters(), {"dpfedavg_norm_bit": False})),
        (ClientProxy("client3"), FitRes(Parameters(), {"dpfedavg_norm_bit": True})),
    ]

    # Execute
    # pylint: disable-next=protected-access
    dp_wrapper._update_clip_norm(results)

    # Assert
    assert dp_wrapper.clip_norm > 0  # Check that clip norm is updated


def test_compute_noise_params() -> None:
    """Test _compute_noise_params method."""
    # Prepare
    noise_multiplier = 1.5
    num_sampled_clients = 5
    clipped_count_stddev = 1.0

    # Execute
    # pylint: disable-next=protected-access
    (
        result_stddev,
        result_multiplier,
    ) = DPStrategyWrapperClientSideAdaptiveClipping._compute_noise_params(
        noise_multiplier, num_sampled_clients, clipped_count_stddev
    )

    # Assert
    assert result_stddev == clipped_count_stddev  # Check stddev consistency
    assert result_multiplier > 0  # Check multiplier consistency
