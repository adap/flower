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
"""Tests for Fashion-MNIST."""


import numpy as np

from .fashion_mnist import shuffle


def test_shuffle() -> None:
    """Test if shuffle is deterministic depending on the the provided seed."""
    # Prepare
    x_tt = np.arange(8)
    y_tt = np.arange(8)

    x_expected_2019 = np.array([1, 4, 3, 6, 7, 5, 2, 0])
    y_expected_2019 = np.array([1, 4, 3, 6, 7, 5, 2, 0])

    x_expected_2020 = np.array([6, 2, 1, 4, 5, 3, 7, 0])
    y_expected_2020 = np.array([6, 2, 1, 4, 5, 3, 7, 0])

    # Execute & assert
    for _ in range(3):
        x_actual, y_actual = shuffle(x_tt, y_tt, seed=2019)
        np.testing.assert_array_equal(x_expected_2019, x_actual)
        np.testing.assert_array_equal(y_expected_2019, y_actual)

    for _ in range(3):
        x_actual, y_actual = shuffle(x_tt, y_tt, seed=2020)
        np.testing.assert_array_equal(x_expected_2020, x_actual)
        np.testing.assert_array_equal(y_expected_2020, y_actual)
