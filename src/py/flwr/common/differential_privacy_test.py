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
"""DP utility functions tests."""

import numpy as np

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays

from .differential_privacy import (
    add_gaussian_noise,
    add_noise_to_params,
    clip_inputs,
    compute_stdv,
    get_norm,
)


def test_add_gaussian_noise() -> None:
    """Test add_gaussian_noise function."""
    # Prepare
    update = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    std_dev = 0.1

    # Execute
    update_noised = add_gaussian_noise(update, std_dev)

    # Assert
    # Check that the shape of the result is the same as the input
    for layer, layer_noised in zip(update, update_noised):
        assert layer.shape == layer_noised.shape

    # Check that the values have been changed and are not equal to the original update
    for layer, layer_noised in zip(update, update_noised):
        assert not np.array_equal(layer, layer_noised)

    # Check that the noise has been added
    for layer, layer_noised in zip(update, update_noised):
        noise_added = layer_noised - layer
        assert np.any(np.abs(noise_added) > 0)


def test_get_norm() -> None:
    """Test get_norm function."""
    # Prepare
    update = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]

    # Execute
    result = get_norm(update)

    expected = float(
        np.linalg.norm(np.concatenate([sub_update.flatten() for sub_update in update]))
    )

    # Assert
    assert expected == result


def test_clip_inputs() -> None:
    """Test clip_inputs function."""
    # Prepare
    updates = [
        np.array([[1.5, -0.5], [2.0, -1.0]]),
        np.array([0.5, -0.5]),
        np.array([[-0.5, 1.5], [-1.0, 2.0]]),
        np.array([-0.5, 0.5]),
    ]
    clipping_norm = 1.5

    # Execute
    clipped_updates = clip_inputs(updates, clipping_norm)

    # Assert
    assert len(clipped_updates) == len(updates)

    for clipped_update, original_update in zip(clipped_updates, updates):
        clip_norm = np.linalg.norm(original_update)
        assert np.all(clipped_update <= clip_norm) and np.all(
            clipped_update >= -clip_norm
        )


def test_add_noise_to_params() -> None:
    """Test add_noise_to_params function."""
    # Prepare
    parameters = ndarrays_to_parameters(
        [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
    )
    std_dev = 0.1

    # Execute
    noised_parameters = add_noise_to_params(parameters, std_dev)

    original_params_list = parameters_to_ndarrays(parameters)
    noised_params_list = parameters_to_ndarrays(noised_parameters)

    # Assert
    assert isinstance(noised_parameters, Parameters)

    # Check the values have been changed and are not equal to the original parameters
    for original_param, noised_param in zip(original_params_list, noised_params_list):
        assert not np.array_equal(original_param, noised_param)

    # Check that the noise has been added
    for original_param, noised_param in zip(original_params_list, noised_params_list):
        noise_added = noised_param - original_param
        assert np.any(np.abs(noise_added) > 0)


def test_compute_stdv() -> None:
    """Test compute_stdv function."""
    # Prepare
    noise_multiplier = 1.0
    clipping_norm = 0.5
    num_sampled_clients = 10

    # Execute
    stdv = compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients)

    # Assert
    expected_stdv = float((noise_multiplier * clipping_norm) / num_sampled_clients)
    assert stdv == expected_stdv
