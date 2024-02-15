# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Utility functions for differential privacy."""


import numpy as np

from flwr.common import (
    NDArrays,
    Parameters,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)


def get_norm(input_arrays: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    array_norms = [np.linalg.norm(array.flat) for array in input_arrays]
    # pylint: disable=consider-using-generator
    return float(np.sqrt(sum([norm**2 for norm in array_norms])))


def add_gaussian_noise_inplace(input_arrays: NDArrays, std_dev: float) -> None:
    """Add Gaussian noise to each element of the input arrays."""
    for array in input_arrays:
        array += np.random.normal(0, std_dev, array.shape)


def clip_inputs_inplace(input_arrays: NDArrays, clipping_norm: float) -> None:
    """Clip model update based on the clipping norm in-place.

    FlatClip method of the paper: https://arxiv.org/abs/1710.06963
    """
    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    for array in input_arrays:
        array *= scaling_factor


def compute_stdv(
    noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    Paper: https://arxiv.org/abs/1710.06963
    """
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)


def compute_clip_model_update(
    param1: NDArrays, param2: NDArrays, clipping_norm: float
) -> None:
    """Compute model update (param1 - param2) and clip it.

    Then add the clipped value to param1."""
    model_update = [np.subtract(x, y) for (x, y) in zip(param1, param2)]
    clip_inputs_inplace(model_update, clipping_norm)

    for i, _ in enumerate(param2):
        param1[i] = param2[i] + model_update[i]


def add_gaussian_noise_to_params(
    model_params: Parameters,
    noise_multiplier: float,
    clipping_norm: float,
    num_sampled_clients: int,
) -> Parameters:
    """Add gaussian noise to model parameters."""
    model_params_ndarrays = parameters_to_ndarrays(model_params)
    add_gaussian_noise_inplace(
        model_params_ndarrays,
        compute_stdv(noise_multiplier, clipping_norm, num_sampled_clients),
    )
    return ndarrays_to_parameters(model_params_ndarrays)
