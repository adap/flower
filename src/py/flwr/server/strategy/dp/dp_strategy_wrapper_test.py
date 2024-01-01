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

from ..fedavg import FedAvg
from .dp_strategy_wrapper import DPWrapper_fixed_clipping


def test_add_gaussian_noise() -> None:
    """Test _add_gaussian_noise function."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPWrapper_fixed_clipping(strategy, 1.5, 1.5, 5)

    update = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    std_dev = 0.1

    # Execute
    update_noised = dp_wrapper._add_gaussian_noise(update, std_dev)

    # Assert
    # Check that the shape of the result is the same as the input
    for layer, layer_noised in zip(update, update_noised):
        assert layer.shape == layer_noised.shape

    # Check that the values have been changed and is not equal to the original update
    assert update != update_noised

    # Check that the noise has been added
    for layer, layer_noised in zip(update, update_noised):
        noise_added = layer_noised - layer
        assert np.any(np.abs(noise_added) > 0)


def test_add_noise_to_updates() -> None:
    """Test _add_noise_to_updates function."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPWrapper_fixed_clipping(strategy, 1.5, 1.5, 5)
    parameters = {"weights": np.array([[1, 2], [3, 4]]), "bias": np.array([0.5, 1.0])}

    # Execute
    result = dp_wrapper._add_noise_to_updates(parameters)

    # Assert
    # Check that the shape of the result is the same as the input params
    for key, value in parameters.items():
        assert value.shape == result[key].shape

    # Check that the values have been changed
    for key, value in parameters.items():
        assert np.any(value != result[key])


def test_get_update_norm() -> None:
    """Test _get_update_norm function."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPWrapper_fixed_clipping(strategy, 1.5, 1.5, 5)
    update = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]

    # Execute
    result = dp_wrapper._get_update_norm(update)

    expected = float(
        np.linalg.norm(np.concatenate([sub_update.flatten() for sub_update in update]))
    )

    # Assert
    assert expected == result


def test_clip_model_updates() -> None:
    """Test _clip_model_updates method."""
    # Prepare
    strategy = FedAvg()
    dp_wrapper = DPWrapper_fixed_clipping(strategy, 1.5, 1.5, 5)

    updates = [
        {
            "weights": np.array([[1.5, -0.5], [2.0, -1.0]]),
            "biases": np.array([0.5, -0.5]),
        },
        {
            "weights": np.array([[-0.5, 1.5], [-1.0, 2.0]]),
            "biases": np.array([-0.5, 0.5]),
        },
    ]

    # Execute
    clipped_updates = dp_wrapper._clip_model_updates(updates)

    # Assert
    assert len(clipped_updates) == len(updates)

    for clipped_update, original_update in zip(clipped_updates, updates):
        for key, clipped_value in clipped_update.items():
            original_value = original_update[key]
            clip_norm = np.linalg.norm(original_value)
            assert np.all(clipped_value <= clip_norm) and np.all(
                clipped_value >= -clip_norm
            )
