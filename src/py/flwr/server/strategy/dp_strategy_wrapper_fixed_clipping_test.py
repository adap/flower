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

from .dp_strategy_wrapper_fixed_clipping import DPStrategyWrapperServerSideFixedClipping
from .fedavg import FedAvg


def test_compute_clip_model_updates() -> None:
    """Test _compute_clip_model_updates method."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPStrategyWrapperServerSideFixedClipping(strategy, 1.5, 10, 5)

    client_params = [
        [
            np.array([0.5, 1.5, 2.5]),
            np.array([3.5, 4.5, 5.5]),
            np.array([6.5, 7.5, 8.5]),
        ],
        [
            np.array([1.5, 2.5, 3.5]),
            np.array([4.5, 5.5, 6.5]),
            np.array([7.5, 8.5, 9.5]),
        ],
    ]
    current_round_params = [
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.array([7.0, 8.0, 9.0]),
    ]

    expected_updates = [
        [
            np.array([0.5, 1.5, 2.5]),
            np.array([3.5, 4.5, 5.5]),
            np.array([6.5, 7.5, 8.5]),
        ],
        [
            np.array([1.5, 2.5, 3.5]),
            np.array([4.5, 5.5, 6.5]),
            np.array([7.5, 8.5, 9.5]),
        ],
    ]
    # Set current model parameters in the wrapper
    dp_wrapper.current_round_params = current_round_params

    # Execute
    dp_wrapper._compute_clip_model_updates(client_params)

    # Verify
    for i, client_param in enumerate(client_params):
        for j, update in enumerate(client_param):
            np.testing.assert_array_almost_equal(update, expected_updates[i][j])


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
        # pylint: disable-next=protected-access
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
