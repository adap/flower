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
"""Tests for partitioned CIFAR-10/100 dataset generation."""
# pylint: disable=no-self-use, invalid-name, disable=R0904

import unittest
from typing import List

import numpy as np
from numpy.random import default_rng

from flwr_baselines.dataset.utils.common import (
    XY,
    combine_partitions,
    create_lda_partitions,
    exclude_classes_and_normalize,
    get_partitions_distributions,
    partition,
    sample_without_replacement,
    shuffle,
    sort_by_label,
    sort_by_label_repeating,
    split_array_at_indices,
    split_at_fraction,
)


def hash_xy(xy: XY) -> int:
    """Return hash of xy."""
    hashes = set()
    for x, y in zip(xy[0], xy[1]):
        hashes.add(hash(x.tobytes() + y.tobytes()))
    return hash(frozenset(hashes))


def assert_identity(xy_0: XY, xy_1: XY) -> None:
    """Assert that both datasets contain the same examples."""
    assert xy_0[0].shape == xy_1[0].shape
    assert xy_0[1].shape == xy_1[1].shape
    assert xy_0[0].dtype == xy_1[0].dtype
    assert xy_0[1].dtype == xy_1[1].dtype
    assert hash_xy(xy_0) == hash_xy(xy_1)


class ImageClassificationPartitionedTestCase(unittest.TestCase):
    """Tests for Partitioned Dataset Generation in Image Classification such as
    CIFAR-10/100."""

    def setUp(self) -> None:
        self.num_classes: int = 10
        self.num_samples_per_class: int = 1000
        self.num_samples: int = self.num_classes * self.num_samples_per_class
        rng = default_rng()
        x = rng.random(size=(self.num_samples, 3, 32, 32))
        y = np.concatenate(
            np.array(
                [self.num_samples_per_class * [j] for j in range(self.num_classes)]
            ),
            axis=0,
        )

        np.random.seed(2000)
        idx = np.random.permutation(x.shape[0])
        x, y = x[idx], y[idx]

        self.ds = x, y

        # Make sure subsequent shuffle in tests
        # produce other permutations
        np.random.seed(3000)

    def test_assert_identity(self) -> None:
        """Test assert_identity function."""
        assert_identity(self.ds, self.ds)

    def test_sort_by_label(self) -> None:
        """Test sort_by_label function."""
        # Prepare
        x_org, y_org = self.ds

        # Execute
        x, y = sort_by_label(x_org, y_org)

        # Assert
        assert_identity(self.ds, (x, y))
        for i, _ in enumerate(y):
            if i > 0:
                assert y[i] >= y[i - 1]

    def test_sort_by_label_repeating(self) -> None:
        """Test sort_by_label function."""
        # Prepare
        x, y = self.ds
        idx = np.random.permutation(x.shape[0])
        x, y = x[idx], y[idx]

        # Execute
        x, y = sort_by_label_repeating(x, y)

        # Assert
        assert_identity(self.ds, (x, y))
        assert set(y[:10]) == set(range(10))

    def test_split_at_fraction(self) -> None:
        """Test split_at_fraction function."""
        # Prepare
        fraction = 0.5
        x, y = self.ds

        # Execute
        (x_0, y_0), (x_1, y_1) = split_at_fraction(x, y, fraction)

        # Assert
        barrier = int(x.shape[0] * fraction)
        np.testing.assert_equal(x_0, x[:barrier])
        np.testing.assert_equal(y_0, y[:barrier])
        np.testing.assert_equal(x_1, x[barrier:])
        np.testing.assert_equal(y_1, y[barrier:])

    def test_shuffle(self) -> None:
        """Test sort_by_label function."""
        # Prepare
        x, y = self.ds

        # Execute
        x, y = shuffle(x, y)

        # Assert
        assert_identity(self.ds, (x, y))

    def test_partition(self) -> None:
        """Test partition function."""
        # Prepare
        x, y = self.ds

        # Execute
        partitions = partition(x, y, 2)

        # Assert
        assert len(partitions) == 2
        assert partitions[0][0].shape == partitions[1][0].shape
        assert partitions[0][1].shape == partitions[1][1].shape

    def test_combine_partitions(self) -> None:
        """Test combine function."""
        # Prepare
        r_0_5 = list(range(0, 5))
        r_5_10 = list(range(5, 10))
        r_0_10 = r_0_5 + r_5_10
        xy_list_0 = [(np.array(r_0_5, np.int64), np.array(r_0_5, np.int64))]
        xy_list_1 = [(np.array(r_5_10, np.int64), np.array(r_5_10, np.int64))]

        # Execute
        xy_combined = combine_partitions(xy_list_0, xy_list_1)

        # Assert
        assert len(xy_combined) == 1
        assert isinstance(xy_combined[0], tuple)
        x_01, y_01 = xy_combined[0]
        np.testing.assert_equal(x_01, r_0_10)
        np.testing.assert_equal(y_01, r_0_10)

    def test_split_array_at_indices_wrong_num_dims(self) -> None:
        """Tests if exception is thrown for wrong number of dimensions."""
        # Prepare
        x = np.ones((100, 3, 32, 32), dtype=np.float32)
        split_idx = np.arange(start=0, stop=90, step=10, dtype=np.int64)
        split_idx = np.expand_dims(split_idx, axis=0)

        # Execute
        with self.assertRaises(ValueError):
            split_array_at_indices(x, split_idx)

    def test_split_array_at_indices_wrong_dtype(self) -> None:
        """Tests if exception is thrown for wrong dtype."""
        # Prepare
        x = np.ones((100, 3, 32, 32), dtype=np.float32)
        split_idx = np.arange(start=0, stop=90, step=10, dtype=np.int32)

        # Execute
        with self.assertRaises(ValueError):
            split_array_at_indices(x, split_idx)

    def test_split_array_at_indices_wrong_split_max_index(self) -> None:
        """Tests if exception is thrown for wrong max split index."""
        # Prepare
        x = np.ones((100, 3, 32, 32), dtype=np.float32)
        split_idx = np.arange(start=0, stop=90, step=10, dtype=np.int64)
        split_idx[-1] = 1000000

        # Execute
        with self.assertRaises(ValueError):
            split_array_at_indices(x, split_idx)

    def test_split_array_at_indices_wrong_initial_split(self) -> None:
        """Tests if exception is thrown for wrong split values."""
        # Prepare
        x = np.ones((100, 3, 32, 32), dtype=np.float32)
        split_idx = np.arange(start=0, stop=90, step=10, dtype=np.int64)
        split_idx[0] = 10

        # Execute
        with self.assertRaises(ValueError):
            split_array_at_indices(x, split_idx)

    def test_split_array_at_indices_not_increasing(self) -> None:
        """Tests if exception is thrown for split not having increasing
        values."""
        # Prepare
        x = np.ones((100, 3, 32, 32), dtype=np.float32)
        split_idx = np.arange(start=0, stop=90, step=10, dtype=np.int64)
        split_idx[1] = 70

        # Execute
        with self.assertRaises(ValueError):
            split_array_at_indices(x, split_idx)

    def test_split_array(self) -> None:
        """Tests if split is correct."""
        # Prepare
        split_expected = [
            [
                np.zeros((3, 32, 32), dtype=np.float32),
                np.zeros((3, 32, 32), dtype=np.float32),
                np.zeros((3, 32, 32), dtype=np.float32),
                np.zeros((3, 32, 32), dtype=np.float32),
            ],
            [
                np.ones((3, 32, 32), dtype=np.float32),
                np.ones((3, 32, 32), dtype=np.float32),
                np.ones((3, 32, 32), dtype=np.float32),
                np.ones((3, 32, 32), dtype=np.float32),
            ],
            [
                2 * np.ones((3, 32, 32), dtype=np.float32),
                2 * np.ones((3, 32, 32), dtype=np.float32),
                2 * np.ones((3, 32, 32), dtype=np.float32),
                2 * np.ones((3, 32, 32), dtype=np.float32),
            ],
        ]

        x = np.concatenate(split_expected)
        split_idx = np.arange(start=0, stop=12, step=4, dtype=np.int64)

        # Execute
        list_splits = split_array_at_indices(x, split_idx)

        # Assert
        for idx, split in enumerate(list_splits):
            for idx_el, element in enumerate(split):
                np.testing.assert_equal(split_expected[idx][idx_el], element)

    def test_exclude_classes_and_normalize_verify_dist_sum_one(self) -> None:
        """Tests if non-distributions raise exceptions."""
        # Prepare
        distribution = np.array([0.1, 0.1, 0.3], dtype=np.float32)
        exclude_dims = 3 * [False]

        # Execute
        with self.assertRaises(ValueError):
            exclude_classes_and_normalize(distribution, exclude_dims=exclude_dims)

    def test_exclude_classes_and_normalize_verify_dist_positive(self) -> None:
        """Tests if non-distributions raise exceptions."""
        # Prepare
        distribution = np.array([0.1, -0.1, 1.0], dtype=np.float32)
        exclude_dims = 3 * [False]

        # Execute
        with self.assertRaises(ValueError):
            exclude_classes_and_normalize(distribution, exclude_dims=exclude_dims)

    def test_exclude_classes_and_normalize_verify_distribution_and_exclude_dims(
        self,
    ) -> None:
        """Tests if non-distributions raise exceptions."""
        # Prepare
        distribution = np.ones((5,), dtype=np.float32)
        exclude_dims = 4 * [False]

        # Execute
        with self.assertRaises(ValueError):
            exclude_classes_and_normalize(distribution, exclude_dims)

    def test_exclude_classes_and_normalize_positive_eps(self) -> None:
        """Tests if eps<0 raises exceptions."""
        # Prepare
        distribution = np.array([0.1, 0.1, 0.8], dtype=np.float32)
        exclude_dims = 3 * [False]

        # Execute
        with self.assertRaises(ValueError):
            exclude_classes_and_normalize(
                distribution, exclude_dims=exclude_dims, eps=-3
            )

    def test_exclude_classes_and_normalize(self) -> None:
        """Tests if non-distributions raise exceptions."""
        # Prepare
        distribution = np.array([0.1, 0.7, 0.1, 0.05, 0.05], dtype=np.float32)
        exclude_dims = [False, True, False, False, False]
        expected = np.array([1.0 / 3, 0, 1.0 / 3, 1.0 / 6, 1.0 / 6], dtype=np.float32)

        # Execute
        new_dist = exclude_classes_and_normalize(distribution, exclude_dims)

        # Assert
        np.testing.assert_array_almost_equal(expected, new_dist, decimal=4)

    def test_sample_without_replacement_large_sample(self) -> None:
        """Tests is requesting too many samples will raise an exception."""
        # Prepare
        distribution = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float32)
        list_samples = [
            [
                np.zeros((3, 32, 32), dtype=np.float32),
                np.zeros((3, 32, 32), dtype=np.float32),
            ],
            [
                np.ones((3, 32, 32), dtype=np.float32),
                np.ones((3, 32, 32), dtype=np.float32),
            ],
            [
                2 * np.ones((3, 32, 32), dtype=np.float32),
                2 * np.ones((3, 32, 32), dtype=np.float32),
            ],
        ]
        num_samples = 100000
        empty_classes = 3 * [False]
        # Execute
        with self.assertRaises(ValueError):
            sample_without_replacement(
                distribution, list_samples, num_samples, empty_classes
            )

    def test_sample_without_replacement_updating_empty_list(self) -> None:
        """Tests is empty list is being updated."""
        # Prepare
        distribution = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=np.float32)
        empty_classes = [False, False, True]
        list_samples: List[List[np.ndarray]] = [
            [
                np.zeros((3, 32, 32), dtype=np.float32),
                np.zeros((3, 32, 32), dtype=np.float32),
            ],
            [
                np.ones((3, 32, 32), dtype=np.float32),
                np.ones((3, 32, 32), dtype=np.float32),
            ],
            [],
        ]
        num_samples = 3
        # Execute
        _, list_empty = sample_without_replacement(
            distribution, list_samples, num_samples, empty_classes
        )

        # Assert
        assert sum(list_empty) == 2

    def test_sample_without_replacement(self) -> None:
        """Tests is sampling is done correctly."""
        # Prepare
        distribution = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        empty_classes = [False, False, True]
        list_samples = [
            [
                np.zeros((3, 1, 1), dtype=np.float32),
                np.zeros((3, 1, 1), dtype=np.float32),
                np.zeros((3, 1, 1), dtype=np.float32),
                np.zeros((3, 1, 1), dtype=np.float32),
                np.zeros((3, 1, 1), dtype=np.float32),
            ],
            [
                7 * np.ones((3, 1, 1), dtype=np.float32),
                7 * np.ones((3, 1, 1), dtype=np.float32),
                7 * np.ones((3, 1, 1), dtype=np.float32),
                7 * np.ones((3, 1, 1), dtype=np.float32),
                7 * np.ones((3, 1, 1), dtype=np.float32),
            ],
            [],
        ]
        num_samples = 4
        expected_x = 7 * np.ones((4, 3, 1, 1), dtype=np.float32)
        expected_y = np.array(4 * [1], dtype=np.int64)

        # Execute
        this_partition, _ = sample_without_replacement(
            distribution, list_samples, num_samples, empty_classes
        )

        # Assert
        assert_identity(this_partition, (expected_x, expected_y))

    def test_create_lda_partitions_imbalanced_not_set(self) -> None:
        """Test if Latent Dirichlet Allocation rejects imbalanced
        partitions."""
        # Prepare
        num_partitions = 3
        concentration = 1e-3

        # Execute
        with self.assertRaises(ValueError):
            create_lda_partitions(
                dataset=self.ds,
                num_partitions=num_partitions,
                concentration=concentration,
            )

    def test_create_lda_partitions_imbalanced(self) -> None:
        """Test if Latent Dirichlet Allocation accepts imbalanced partitions if
        accept_imbalanced is set."""
        # Prepare
        num_partitions = 3
        concentration = 1e-3

        # Execute
        partitions, _ = create_lda_partitions(
            dataset=self.ds,
            num_partitions=num_partitions,
            concentration=concentration,
            accept_imbalanced=True,
        )
        numel_list = [x.shape[0] for (x, y) in partitions]
        total_samples = np.sum(numel_list)

        # Assert
        assert total_samples == self.num_samples

    def test_create_lda_partitions_alpha_near_zero(self) -> None:
        """Test if Latent Dirichlet Allocation partitions will give single
        class distribution when concentration is near zero (~1e-3)."""
        # Prepare
        num_partitions = 5
        concentration = 1e-3

        # Execute
        _, distributions = create_lda_partitions(
            dataset=self.ds, num_partitions=num_partitions, concentration=concentration
        )
        test_num_partitions, _ = distributions.shape

        # Assert
        for part in range(test_num_partitions):
            this_distribution = distributions[part]
            max_prob = np.max(this_distribution)
            assert max_prob > 0.5

    def test_create_lda_partitions_large_alpha(self) -> None:
        """Test if Latent Dirichlet Allocation partitions will give near
        uniform distribution when concentration is large(~1e5)."""
        # Prepare
        num_partitions = 5
        concentration = 1e5
        uniform = (
            1.0 / (self.num_classes) * np.ones((self.num_classes,), dtype=np.float64)
        )

        # Execute
        _, distributions = create_lda_partitions(
            dataset=self.ds, num_partitions=num_partitions, concentration=concentration
        )
        test_num_partitions, _ = distributions.shape

        # Assert
        for part in range(test_num_partitions):
            this_distribution = distributions[part]
            np.testing.assert_array_almost_equal(this_distribution, uniform, decimal=3)

    def test_get_partitions_distributions(self) -> None:
        """Tests if function can generate data pdf for each partition."""
        # Prepare
        partitions = [
            (np.zeros((2, 5, 5), dtype=np.float32), np.arange(2)),
            (np.zeros((5, 5, 5), dtype=np.float32), np.arange(5)),
            (np.zeros((3, 5, 5), dtype=np.float32), np.arange(3)),
        ]

        # Execute
        distributions, labels = get_partitions_distributions(partitions)

        # Assert
        np.testing.assert_array_equal(
            distributions[0], np.array([0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            distributions[1], np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
        )
        np.testing.assert_array_equal(
            distributions[2],
            np.array([1.0 / 3, 1.0 / 3, 1.0 / 3, 0.0, 0.0], dtype=np.float32),
        )
        assert labels == [0, 1, 2, 3, 4]

    def test_create_lda_partitions_elements(self) -> None:
        """Test if partitions from Latent Dirichlet Allocation contain the same
        elements."""
        # Prepare
        num_partitions = 5
        concentration = 0.5

        # Execute
        partitions, _ = create_lda_partitions(
            dataset=self.ds, num_partitions=num_partitions, concentration=concentration
        )
        x_lda = np.concatenate([item[0] for item in partitions])
        y_lda = np.concatenate([item[1] for item in partitions])

        # Assert
        assert_identity(xy_0=self.ds, xy_1=(x_lda, y_lda))

    def test_create_lda_partitions_with_inf_alpha(self) -> None:
        """Test if partitions created with concentration=Inf will produce
        uniform partitions."""
        # Prepare
        num_partitions = 5
        concentration = float("inf")

        # Execute
        partitions, dirichlet_dist = create_lda_partitions(
            dataset=self.ds, num_partitions=num_partitions, concentration=concentration
        )
        x_lda = np.concatenate([item[0] for item in partitions])
        y_lda = np.concatenate([item[1] for item in partitions])

        # Assert
        np.testing.assert_array_equal(
            dirichlet_dist,
            1.0
            / self.num_classes
            * np.ones((num_partitions, self.num_classes), dtype=np.float32),
        )
        assert_identity(xy_0=self.ds, xy_1=(x_lda, y_lda))

    def test_create_lda_partitions_elements_list_concentration(self) -> None:
        """Test if partitions from Latent Dirichlet Allocation contain the same
        elements."""
        # Prepare
        num_partitions = 5
        concentration = self.num_classes * [0.5]

        # Execute
        partitions, _ = create_lda_partitions(
            dataset=self.ds, num_partitions=num_partitions, concentration=concentration
        )
        x_lda = np.concatenate([item[0] for item in partitions])
        y_lda = np.concatenate([item[1] for item in partitions])

        # Assert
        assert_identity(xy_0=self.ds, xy_1=(x_lda, y_lda))

    def test_create_lda_partitions_elements_wrong_list_concentration(self) -> None:
        """Test if partitions from Latent Dirichlet Allocation contain the same
        elements."""
        # Prepare
        num_partitions = 5
        concentration = (self.num_classes + 1) * [0.5]

        # Execute
        with self.assertRaises(ValueError):
            create_lda_partitions(
                dataset=self.ds,
                num_partitions=num_partitions,
                concentration=concentration,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
