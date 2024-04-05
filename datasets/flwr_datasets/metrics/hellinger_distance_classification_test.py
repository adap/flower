# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""HellingerDistanceClassification tests."""

import string
import unittest

import numpy as np

from flwr_datasets.metrics.hellinger_distance_classification import (
    HellingerDistanceClassification,
)


class TestHellingerDistanceClassification(unittest.TestCase):
    """Test HellingerDistanceClassification."""

    def setUp(self) -> None:
        """Set up values."""
        # Number of clients to create
        self.n_clients = 10
        # Number of examples per client
        self.n_examples = 10000
        # Number of different classes (categories)
        self.n_class = 20
        # Seed for reproducibility
        self.random_state = 42

    def test_edge_case_lowest_distance(self) -> None:
        """Test if the distance is equal to 0 when each client has the same targets."""
        dummy_targets_lists = [
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
        ]

        HD = HellingerDistanceClassification(dummy_targets_lists)

        self.assertEqual(HD, 0)

    def test_edge_case_highest_distance(self) -> None:
        """Test if the distance is equal to 1 when each client has disjoint targets."""
        dummy_targets_lists = [
            [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6],
        ]

        HD = HellingerDistanceClassification(dummy_targets_lists)

        self.assertEqual(HD, 1)

    def test_correctness_low_non_IID_ness(self) -> None:
        """Test if the distance matches the manually computed distance for low non-IID-
        ness case.
        """
        dummy_targets_lists = [
            [11, 0, 9, 9, 0, 9, 11, 23, 9, 11, 12, 23, 6, 0, 9, 5, 8],
            [0, 13, 9, 3, 11, 11, 5, 7, 9, 10, 9, 8, 15, 12, 11, 8, 11, 2, 11, 9, 0],
        ]

        HD = HellingerDistanceClassification(dummy_targets_lists)

        self.assertEqual(HD, 0.49798530303593314)

    def test_correctness_high_non_IID_ness(self) -> None:
        """Test if the distance matches the manually computed distance for high non-IID-
        ness case.
        """
        dummy_targets_lists = [
            [24, 25, 9, 26, 8, 12, 24, 24, 25, 12, 12, 12, 22, 8, 24, 24],
            [11, 11, 5, 3, 3, 11, 11, 3, 11, 11, 3, 7, 5, 11, 11, 11, 7, 5],
            [3, 5, 10, 17, 2, 2, 3, 2, 3, 2, 17, 17, 0, 2],
            [9, 9, 9, 9, 14, 1, 19, 9, 9, 9, 9, 19, 14, 9, 19, 9, 23],
        ]

        HD = HellingerDistanceClassification(dummy_targets_lists)

        self.assertEqual(HD, 0.9557660032171551)

    def test_target_integer(self) -> None:
        """Test if the function works well with integer input."""
        np.random.seed(self.random_state)
        random_targets_lists = np.random.randint(
            low=0, high=self.n_class, size=(self.n_clients, self.n_examples)
        ).tolist()

        HD = HellingerDistanceClassification(random_targets_lists)

        self.assertTrue(0 <= HD <= 1)

    def test_target_string(self) -> None:
        """Test if the function works well with string input."""
        np.random.seed(self.random_state)
        random_targets_lists = np.random.choice(
            list(string.ascii_lowercase), size=(self.n_clients, self.n_examples)
        )

        HD = HellingerDistanceClassification(random_targets_lists)

        self.assertTrue(0 <= HD <= 1)

    def test_target_bool(self) -> None:
        """Test if the function works well with boolean input."""
        np.random.seed(self.random_state)
        random_targets_lists = np.random.choice(
            a=[False, True], size=(self.n_clients, self.n_examples)
        ).tolist()

        HD = HellingerDistanceClassification(random_targets_lists)

        self.assertTrue(0 <= HD <= 1)


if __name__ == "__main__":
    unittest.main()
