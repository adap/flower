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


def test_compute_model_updates() -> None:
    """Test _compute_model_updates method."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPStrategyWrapperServerSideFixedClipping(strategy, 1.5, 1.5, 5)

    client_params = [
        [np.array([2, 3, 4]), np.array([5, 6, 7])],
        [np.array([3, 4, 5]), np.array([6, 7, 8])],
    ]
    current_round_params = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    expected_updates = [
        [
            np.subtract(client_params[0][0], current_round_params[0]),
            np.subtract(client_params[0][1], current_round_params[1]),
        ],
        [
            np.subtract(client_params[1][0], current_round_params[0]),
            np.subtract(client_params[1][1], current_round_params[1]),
        ],
    ]
    # Set current model parameters in the wrapper
    dp_wrapper.current_round_params = current_round_params

    # Execute
    computed_updates = dp_wrapper._compute_model_updates(client_params)

    for expected, actual in zip(expected_updates, computed_updates):
        for e, a in zip(expected, actual):
            np.testing.assert_array_equal(e, a)


def test_update_clients_params() -> None:
    """Test _update_clients_params method."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPStrategyWrapperServerSideFixedClipping(strategy, 1.5, 1.5, 5)

    client_params = [
        [np.array([2, 3, 4]), np.array([5, 6, 7])],
        [np.array([3, 4, 5]), np.array([6, 7, 8])],
    ]
    client_update = [
        [np.array([1, 1, 1]), np.array([1, 1, 1])],
        [np.array([1, 1, 1]), np.array([1, 1, 1])],
    ]
    current_round_params = [np.array([1, 2, 3]), np.array([4, 5, 6])]

    # Set current model parameters in the wrapper
    dp_wrapper.current_round_params = current_round_params

    # Execute
    for params, update in zip(client_params, client_update):
        dp_wrapper._update_clients_params(params, update)

    # Assert
    expected_params = [
        [
            np.add(current_round_params[0], client_update[0][0]),
            np.add(current_round_params[1], client_update[0][1]),
        ],
        [
            np.add(current_round_params[0], client_update[1][0]),
            np.add(current_round_params[1], client_update[1][1]),
        ],
    ]

    for expected, actual in zip(expected_params, client_params):
        for e, a in zip(expected, actual):
            np.testing.assert_array_equal(e, a)
