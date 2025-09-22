# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for Bulyan."""


import numpy as np
import pytest

from .bulyan import aggregate_n_closest_weights


def test_single_layer_single_value() -> None:
    """Trivial case: one layer, one value in each weight."""
    ref = [np.array([1.0])]
    weights = [[np.array([1.0])], [np.array([2.0])], [np.array([0.5])]]
    result = aggregate_n_closest_weights(ref, weights, beta=2)
    # Closest to 1.0 are 1.0 and 0.5 -> mean = 0.75
    np.testing.assert_allclose(result[0], np.array([0.75]))


def test_multiple_layers() -> None:
    """Two layers with different shapes."""
    ref = [np.array([1.0, 2.0]), np.array([[0.0, 1.0]])]
    weights = [
        [np.array([1.1, 2.1]), np.array([[0.2, 1.2]])],
        [np.array([0.9, 1.9]), np.array([[-0.1, 0.8]])],
        [np.array([5.0, 5.0]), np.array([[10.0, 10.0]])],
    ]
    result = aggregate_n_closest_weights(ref, weights, beta=2)

    # First layer: closest to [1.0,2.0] are [1.1,2.1] and [0.9,1.9] -> mean = [1.0,2.0]
    np.testing.assert_allclose(result[0], np.array([1.0, 2.0]))
    # Second layer: closest are [0.2,1.2] and [-0.1,0.8] -> mean = [0.05,1.0]
    np.testing.assert_allclose(result[1], np.array([[0.05, 1.0]]))


def test_beta_equals_one() -> None:
    """When beta=1, result should be the single closest weight coordinate-wise."""
    ref = [np.array([0.0])]
    weights = [[np.array([1.0])], [np.array([-0.5])], [np.array([10.0])]]
    result = aggregate_n_closest_weights(ref, weights, beta=1)
    # Closest to 0 is -0.5 (distance 0.5 vs 1.0 vs 10.0)
    np.testing.assert_allclose(result[0], np.array([-0.5]))


def test_beta_equals_all() -> None:
    """When beta equals number of models, it should just average all weights."""
    ref = [np.array([0.0, 0.0])]
    weights = [[np.array([1.0, 2.0])], [np.array([-1.0, 0.0])], [np.array([3.0, 4.0])]]
    result = aggregate_n_closest_weights(ref, weights, beta=3)
    expected = np.mean([w[0] for w in weights], axis=0)
    np.testing.assert_allclose(result[0], expected)


def test_nontrivial_shapes() -> None:
    """Handles higher-dimensional arrays correctly."""
    ref = [np.array([[1.0, 2.0], [3.0, 4.0]])]
    weights = [
        [np.array([[1.1, 2.1], [2.9, 3.9]])],
        [np.array([[10.0, 10.0], [10.0, 10.0]])],
        [np.array([[0.9, 1.9], [3.1, 4.1]])],
    ]
    result = aggregate_n_closest_weights(ref, weights, beta=2)
    expected = np.mean([weights[0][0], weights[2][0]], axis=0)  # ignore the far one
    np.testing.assert_allclose(result[0], expected)


def test_invalid_beta_too_large() -> None:
    """Should raise if beta > number of available weights."""
    ref = [np.array([1.0])]
    weights = [[np.array([1.0])], [np.array([2.0])]]
    with pytest.raises(ValueError):
        aggregate_n_closest_weights(ref, weights, beta=3)


def test_identical_weights() -> None:
    """If all candidate weights are identical, result should equal that weight."""
    ref = [np.array([42.0, -1.0])]
    weights = [[np.array([5.0, 5.0])], [np.array([5.0, 5.0])], [np.array([5.0, 5.0])]]
    result = aggregate_n_closest_weights(ref, weights, beta=2)
    np.testing.assert_allclose(result[0], np.array([5.0, 5.0]))
