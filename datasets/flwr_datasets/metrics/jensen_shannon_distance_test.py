import string
import unittest

import numpy as np
from flower.datasets.flwr_datasets.metrics.jensen_shannon_distance import (
    JensenShannonDistance,
)


class TestJensenShannonDistance(unittest.TestCase):
    """Test JensenShannonDistance."""

    def setUp(self):
        # Number of clients to create
        self.n_clients = 10
        # Number of examples per client
        self.n_examples = 10000
        # Number of different classes (categories)
        self.n_class = 20
        # Seed for reproducibility
        self.random_state = 42

    def test_correctness_low_non_IID_ness(self):
        """Test if the distance matches the manually computed distance for low non-IID-
        ness case.
        """
        dummy_targets_lists = [
            [11, 0, 9, 9, 0, 9, 11, 23, 9, 11, 12, 23, 6, 0, 9, 5, 8],
            [0, 13, 9, 3, 11, 11, 5, 7, 9, 10, 9, 8, 15, 12, 11, 8, 11, 2, 11, 9, 0],
        ]

        JSD = JensenShannonDistance().calculate(dummy_targets_lists)

        self.assertEqual(JSD, 0.5053181562661684)

    def test_correctness_high_non_IID_ness(self):
        """Test if the distance matches the manually computed distance for high non-IID-
        ness case.
        """
        dummy_targets_lists = [
            [24, 25, 9, 26, 8, 12, 24, 24, 25, 12, 12, 12, 22, 8, 24, 24],
            [11, 11, 5, 3, 3, 11, 11, 3, 11, 11, 3, 7, 5, 11, 11, 11, 7, 5],
            [3, 5, 10, 17, 2, 2, 3, 2, 3, 2, 17, 17, 0, 2],
            [9, 9, 9, 9, 14, 1, 19, 9, 9, 9, 9, 19, 14, 9, 19, 9, 23],
        ]

        JSD = JensenShannonDistance().calculate(dummy_targets_lists)

        self.assertEqual(JSD, 0.9392026840151618)

    def test_target_integer(self):
        """Test if the function works well with integer input."""
        np.random.seed(self.random_state)
        random_targets_lists = list(
            np.random.randint(
                low=0, high=self.n_class, size=(self.n_clients, self.n_examples)
            )
        )

        JSD = JensenShannonDistance().calculate(random_targets_lists)

        self.assertTrue(0 <= JSD <= 1)

    def test_target_float(self):
        """Test if the function works well with float input."""
        np.random.seed(self.random_state)
        random_targets_lists = list(
            np.random.uniform(
                low=0, high=self.n_class, size=(self.n_clients, self.n_examples)
            )
        )

        JSD = JensenShannonDistance().calculate(random_targets_lists)

        self.assertTrue(0 <= JSD <= 1)

    def test_target_string(self):
        """Test if the function works well with string input."""
        np.random.seed(self.random_state)
        random_targets_lists = np.random.choice(
            list(string.ascii_lowercase), size=(self.n_clients, self.n_examples)
        )

        JSD = JensenShannonDistance().calculate(random_targets_lists)

        self.assertTrue(0 <= JSD <= 1)

    def test_target_regression_task(self):
        """Test if the function works well changing the task_type to regression."""
        np.random.seed(self.random_state)
        random_targets_lists = list(
            np.random.uniform(
                low=0, high=self.n_class, size=(self.n_clients, self.n_examples)
            )
        )

        JSD = JensenShannonDistance(task_type="regression", num_bins=10).calculate(
            random_targets_lists
        )
        self.assertTrue(0 <= JSD <= 1)


if __name__ == "__main__":
    unittest.main()
