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
"""Utility functions for performing operations on Numpy NDArrays."""


from typing import List, Tuple

import numpy as np


def factor_weights_combine(
    weights_factor: int, weights: List[np.ndarray]
) -> List[np.ndarray]:
    """Combine weight factor with weights."""
    return [np.array([weights_factor])] + weights


def factor_weights_extract(weights: List[np.ndarray]) -> Tuple[int, List[np.ndarray]]:
    """Extract weight factor from weights."""
    return weights[0][0], weights[1:]


def weights_shape(weights: List[np.ndarray]) -> List[Tuple]:
    """Get dimensions of each NDArray in weights."""
    return [arr.shape for arr in weights]


def weights_zero_generate(
    dimensions_list: List[Tuple], dtype=np.int64
) -> List[np.ndarray]:
    """Generate zero weights based on the dimensions list."""
    return [np.zeros(dimensions, dtype=dtype) for dimensions in dimensions_list]


def weights_addition(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    """Add two weights."""
    return [a[idx] + b[idx] for idx in range(len(a))]


def weights_subtraction(a: List[np.ndarray], b: List[np.ndarray]) -> List[np.ndarray]:
    """Subtract one weight from the other."""
    return [a[idx] - b[idx] for idx in range(len(a))]


def weights_mod(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    """Take mod of a weights with an integer."""
    if bin(b).count("1") == 1:
        msk = b - 1
        return [a[idx] & msk for idx in range(len(a))]
    return [a[idx] % b for idx in range(len(a))]


def weights_multiply(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    """Multiply weight by an integer."""
    return [a[idx] * b for idx in range(len(a))]


def weights_divide(a: List[np.ndarray], b: int) -> List[np.ndarray]:
    """Divide weight by an integer."""
    return [a[idx] / b for idx in range(len(a))]
