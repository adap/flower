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
"""dp_strategy_wrapper_fixed_clipping tests."""

import numpy as np

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from .dp_strategy_wrapper_fixed_clipping import DPStrategyWrapperServerSideFixedClipping
from .fedavg import FedAvg


def test_add_noise_to_updates() -> None:
    """Test _add_noise_to_updates method."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPStrategyWrapperServerSideFixedClipping(strategy, 1.5, 1.5, 5)
    parameters = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]

    # Execute
    result = parameters_to_ndarrays(
        # pylint: disable-next=protected-access
        dp_wrapper._add_noise_to_updates(ndarrays_to_parameters(parameters))
    )

    # Assert
    for layer in result:
        assert layer.shape == parameters[0].shape  # Check shape consistency
        assert not np.array_equal(layer, parameters[0])  # Check if noise was added
