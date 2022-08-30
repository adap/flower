# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Building block functions for DP algorithms."""

from typing import Tuple

import numpy as np

from flwr.common.typing import NDArrays


# Calculates the L2-norm of a potentially ragged array
def _get_update_norm(update: NDArrays) -> float:
    flattened_update = update[0]
    for i in range(1, len(update)):
        flattened_update = np.append(flattened_update, update[i])  # type: ignore
    return float(np.sqrt(np.sum(np.square(flattened_update))))


def add_gaussian_noise(update: NDArrays, std_dev: float) -> NDArrays:
    """Adds iid Gaussian noise of the given standard deviation to each floating
    point value in the update."""
    update_noised = [
        layer + np.random.normal(0, std_dev, layer.shape) for layer in update
    ]
    return update_noised


def clip_by_l2(update: NDArrays, threshold: float) -> Tuple[NDArrays, bool]:
    """Scales the update so thats its L2 norm is upper-bound to threshold."""
    update_norm = _get_update_norm(update)
    scaling_factor = min(1, threshold / update_norm)
    update_clipped: NDArrays = [layer * scaling_factor for layer in update]
    return update_clipped, (scaling_factor < 1)
