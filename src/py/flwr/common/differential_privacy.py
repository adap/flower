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


def get_norm(input_array: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    flattened_input = np.concatenate(
        [np.asarray(sub_input).flatten() for sub_input in input_array]
    )
    return float(np.linalg.norm(flattened_input))


def add_gaussian_noise(input_array: NDArrays, std_dev: float) -> NDArrays:
    """Add noise to each element of the provided input from Gaussian (Normal)
    distribution with respect to the passed standard deviation."""
    noised_input = [
        layer + np.random.normal(0, std_dev, layer.shape) for layer in input_array
    ]
    return noised_input


def clip_inputs(input_array: NDArrays, clipping_norm: float) -> NDArrays:
    """Clip model update based on the clipping norm.

    FlatClip method of the paper: https://arxiv.org/pdf/1710.06963.pdf
    """
    input_norm = get_norm(input_array)
    scaling_factor = min(1, clipping_norm / input_norm)
    clipped_inputs: NDArrays = [layer * scaling_factor for layer in input_array]
    return clipped_inputs


def add_noise_to_params(parameters: Parameters, stdv: float) -> Parameters:
    """Add Gaussian noise to model params."""
    return ndarrays_to_parameters(
        add_gaussian_noise(
            parameters_to_ndarrays(parameters),
            stdv,
        )
    )


def compute_stdv(
    noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    paper: https://arxiv.org/pdf/1710.06963.pdf
    """
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)
