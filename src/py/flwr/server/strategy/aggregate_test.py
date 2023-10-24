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

from .aggregate import aggregate, aggregate_meamed, weighted_loss_avg


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


def test_aggregate_meamed() -> None:
    """Test mean around median aggregation."""
    weights0 = np.array([[1, 6, 11], [16, 21, 26]])
    weights1 = np.array([[2, 7, 12], [17, 22, 27]])
    weights3 = np.array([[3, 8, 13], [18, 23, 28]])
    weights4 = np.array([[4, 9, 14], [19, 24, 29]])
    weights5 = np.array([[5, 10, 15], [20, 25, 30]])

    results = [
        (weights0, 1),
        (weights1, 1),
        (weights3, 1),
        (weights4, 1),
        (weights5, 1),
    ]
    expected = [np.array([3.0, 8.0, 13.0]), np.array([18.0, 23.0, 28.0])]

    # Execute
    actual = aggregate_meamed(results, 2)

    # Assert
    np.testing.assert_equal(expected, actual)  # type: ignore
