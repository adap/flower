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


def get_norm(input: NDArrays) -> float:
    """Compute the L2 norm of the flattened input."""
    flattened_input = np.concatenate(
        [np.asarray(sub_input).flatten() for sub_input in input]
    )
    return float(np.linalg.norm(flattened_input))


def add_gaussian_noise(input: NDArrays, std_dev: float) -> NDArrays:
    """Add noise to each element of the provided input from Gaussian (Normal)
    distribution with respect to the passed standard deviation."""
    noised_input = [
        layer + np.random.normal(0, std_dev, layer.shape) for layer in input
    ]
    return noised_input

def clip_inputs(self, update: NDArrays, ) -> NDArrays:
    """Clip model update based on the computed clipping_threshold.

    FlatClip method of the paper: https://arxiv.org/pdf/1710.06963.pdf
    """
    update_norm = get_norm(update)
    scaling_factor = min(1, self.clipping_threshold / update_norm)
    update_clipped: NDArrays = [layer * scaling_factor for layer in update]
    return update_clipped
