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

from flwr.common import NDArrays


def get_norm(input_arrays: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    array_norms = [np.linalg.norm(array.flat) for array in input_arrays]
    return float(
        np.sqrt(sum([norm**2 for norm in array_norms]))
    )  # pylint: disable=consider-using-generator


def add_gaussian_noise_inplace(input_arrays: NDArrays, std_dev: float) -> None:
    """Add noise to each element of the provided input from Gaussian (Normal)
    distribution with respect to the passed standard deviation."""
    for array in input_arrays:
        array += np.random.normal(0, std_dev, array.shape).astype(array.dtype)


def clip_inputs(input_arrays: NDArrays, clipping_norm: float) -> NDArrays:
    """Clip model update based on the clipping norm.

    FlatClip method of the paper: https://arxiv.org/pdf/1710.06963.pdf
    """
    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    clipped_inputs: NDArrays = [layer * scaling_factor for layer in input_arrays]
    return clipped_inputs


def clip_inputs_inplace(input_arrays: NDArrays, clipping_norm: float) -> None:
    """Clip model update based on the clipping norm in-place.

    FlatClip method of the paper: https://arxiv.org/pdf/1710.06963.pdf
    """
    input_norm = get_norm(input_arrays)
    scaling_factor = min(1, clipping_norm / input_norm)
    for array in input_arrays:
        array *= scaling_factor


def compute_stdv(
    noise_multiplier: float, clipping_norm: float, num_sampled_clients: int
) -> float:
    """Compute standard deviation for noise addition.

    paper: https://arxiv.org/pdf/1710.06963.pdf
    """
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)
