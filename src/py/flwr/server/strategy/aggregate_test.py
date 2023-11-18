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
"""Aggregation function tests."""


from typing import List, Tuple

import numpy as np

from .aggregate import (
    _aggregate_n_closest_weights,
    _check_weights_equality,
    _find_reference_weights,
    aggregate,
    weighted_loss_avg,
)


def test_aggregate() -> None:
    """Test aggregate function."""
    # Prepare
    weights0_0 = np.array([[1, 2, 3], [4, 5, 6]])
    weights0_1 = np.array([7, 8, 9, 10])
    weights1_0 = np.array([[1, 2, 3], [4, 5, 6]])
    weights1_1 = np.array([7, 8, 9, 10])
    results = [([weights0_0, weights0_1], 1), ([weights1_0, weights1_1], 2)]

    expected = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([7, 8, 9, 10])]

    # Execute
    actual = aggregate(results)

    # Assert
    np.testing.assert_equal(expected, actual)


def test_weighted_loss_avg_single_value() -> None:
    """Test weighted loss averaging."""
    # Prepare
    results: List[Tuple[int, float]] = [(5, 0.5)]
    expected = 0.5

    # Execute
    actual = weighted_loss_avg(results)

    # Assert
    assert expected == actual


def test_weighted_loss_avg_multiple_values() -> None:
    """Test weighted loss averaging."""
    # Prepare
    results: List[Tuple[int, float]] = [(1, 2.0), (2, 1.0), (1, 2.0)]
    expected = 1.5

    # Execute
    actual = weighted_loss_avg(results)

    # Assert
    assert expected == actual


def test_check_weights_equality_true() -> None:
    """Check weights equality - the same weights."""
    weights1 = [np.array([1, 2]), np.array([[1, 2], [3, 4]])]
    weights2 = [np.array([1, 2]), np.array([[1, 2], [3, 4]])]
    results = _check_weights_equality(weights1, weights2)
    expected = True
    assert expected == results


def test_check_weights_equality_numeric_false() -> None:
    """Check weights equality - different weights, same length."""
    weights1 = [np.array([1, 2]), np.array([[1, 2], [3, 4]])]
    weights2 = [np.array([2, 2]), np.array([[1, 2], [3, 4]])]
    results = _check_weights_equality(weights1, weights2)
    expected = False
    assert expected == results


def test_check_weights_equality_various_length_false() -> None:
    """Check weights equality - the same first layer weights, different length."""
    weights1 = [np.array([1, 2]), np.array([[1, 2], [3, 4]])]
    weights2 = [np.array([1, 2])]
    results = _check_weights_equality(weights1, weights2)
    expected = False
    assert expected == results


def test_find_reference_weights() -> None:
    """Check if the finding weights from list of weigths work."""
    reference_weights = [np.array([1, 2]), np.array([[1, 2], [3, 4]])]
    list_of_weights = [
        [np.array([2, 2]), np.array([[1, 2], [3, 4]])],
        [np.array([3, 2]), np.array([[1, 2], [3, 4]])],
        [np.array([3, 2]), np.array([[1, 2], [10, 4]])],
        [np.array([1, 2]), np.array([[1, 2], [3, 4]])],
    ]

    result = _find_reference_weights(reference_weights, list_of_weights)

    expected = 3
    assert result == expected


def test_aggregate_n_closest_weights_mean() -> None:
    """Check if aggregation of n closest weights to the reference works."""
    beta_closest = 2
    reference_weights = [np.array([1, 2]), np.array([[1, 2], [3, 4]])]

    list_of_weights = [
        [np.array([1, 2]), np.array([[1, 2], [3, 4]])],
        [np.array([1.1, 2.1]), np.array([[1.1, 2.1], [3.1, 4.1]])],
        [np.array([1.2, 2.2]), np.array([[1.2, 2.2], [3.2, 4.2]])],
        [np.array([1.3, 2.3]), np.array([[0.9, 2.5], [3.4, 3.8]])],
    ]
    list_of_weights_and_samples = [(weights, 0) for weights in list_of_weights]

    beta_closest_weights = _aggregate_n_closest_weights(
        reference_weights, list_of_weights_and_samples, beta_closest=beta_closest
    )
    expected_averaged = [np.array([1.05, 2.05]), np.array([[0.95, 2.05], [3.05, 4.05]])]

    assert all(
        (
            np.array_equal(expected, result)
            for expected, result in zip(expected_averaged, beta_closest_weights)
        )
    )
