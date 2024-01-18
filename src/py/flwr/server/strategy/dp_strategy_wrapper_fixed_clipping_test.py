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
"""DPWrapper_fixed_clipping tests."""

import numpy as np

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays

from .dp_strategy_wrapper_fixed_clipping import DPStrategyWrapperServerSideFixedClipping
from .fedavg import FedAvg


def test_add_gaussian_noise() -> None:
    """Test add_gaussian_noise function."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPStrategyWrapperServerSideFixedClipping(strategy, 1.5, 1.5, 5)

    update = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    std_dev = 0.1

    # Execute
    update_noised = dp_wrapper.add_gaussian_noise(update, std_dev)

    # Assert
    # Check that the shape of the result is the same as the input
    for layer, layer_noised in zip(update, update_noised):
        assert layer.shape == layer_noised.shape

    # Check that the values have been changed and is not equal to the original update
    for layer, layer_noised in zip(update, update_noised):
        assert not np.array_equal(layer, layer_noised)

    # Check that the noise has been added
    for layer, layer_noised in zip(update, update_noised):
        noise_added = layer_noised - layer
        assert np.any(np.abs(noise_added) > 0)


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


def test_get_update_norm() -> None:
    """Test get_update_norm function."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPStrategyWrapperServerSideFixedClipping(strategy, 1.5, 1.5, 5)
    update = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]

    # Execute
    result = dp_wrapper.get_update_norm(update)

    expected = float(
        np.linalg.norm(np.concatenate([sub_update.flatten() for sub_update in update]))
    )

    # Assert
    assert expected == result


def test_clip_model_updates() -> None:
    """Test _clip_model_updates method."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPStrategyWrapperServerSideFixedClipping(strategy, 1.5, 1.5, 5)

    updates = [
        np.array([[1.5, -0.5], [2.0, -1.0]]),
        np.array([0.5, -0.5]),
        np.array([[-0.5, 1.5], [-1.0, 2.0]]),
        np.array([-0.5, 0.5]),
    ]

    # Execute
    # pylint: disable-next=protected-access
    clipped_updates = dp_wrapper._clip_model_update(updates)

    # Assert
    assert len(clipped_updates) == len(updates)

    for clipped_update, original_update in zip(clipped_updates, updates):
        clip_norm = np.linalg.norm(original_update)
        assert np.all(clipped_update <= clip_norm) and np.all(
            clipped_update >= -clip_norm
        )
