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
"""Tests for partitioned CIFAR-10/100 dataset generation."""
# pylint: disable=no-self-use, invalid-name

import unittest

import numpy as np

from flwr.dataset.utils.common import (
    XY,
    combine_partitions,
    create_dla_partitions,
    partition,
    shuffle,
    sort_by_label,
    sort_by_label_repeating,
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
    """Tests for Partitioned Dataset Ggeneration in Image Classification such
    as CIFAR-10/100."""

    def setUp(self) -> None:
        self.num_classes: int = 10
        self.num_samples_per_class: int = 1000
        self.num_samples: int = self.num_classes * self.num_samples_per_class
        x = np.random.random(size=(self.num_samples, 3, 32, 32))
        y = np.concatenate(
            np.array(
                [self.num_samples_per_class * [j] for j in range(self.num_classes)]
            ),
            axis=0,
        )
        y = np.expand_dims(y, axis=1)

        np.random.seed(2000)
        idx = np.random.permutation(x.shape[0])
        x, y = x[idx], y[idx]

        self.ds = x, y

        # Make sure subsequent shuffle in tests
        # produce other permuations
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
        assert {y[0] for y in y[:10]} == set(range(10))

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

    def test_create_dla_partitions_alpha_near_zero(self) -> None:
        """Test if Dirichlet Latent Allocation partitions will give single
        class distribution when concentration is near zero (~1e-3)."""
        # Prepare
        num_partitions = 5
        concentration = 1e-3

        # Execute
        _, distributions = create_dla_partitions(
            dataset=self.ds, num_partitions=num_partitions, concentration=concentration
        )
        test_num_partitions, _ = distributions.shape

        # Assert
        for part in range(test_num_partitions):
            this_distribution = distributions[part]
            max_prob = np.max(this_distribution)
            assert max_prob > 0.5

    def test_create_dla_partitions_large_alpha(self) -> None:
        """Test if Dirichlet Latent Allocation partitions will give near
        uniform distribution when concentration is large(~1e5)."""
        # Prepare
        num_partitions = 5
        concentration = 1e5
        uniform = (
            1.0 / (self.num_classes) * np.ones((self.num_classes,), dtype=np.float)
        )

        # Execute
        _, distributions = create_dla_partitions(
            dataset=self.ds, num_partitions=num_partitions, concentration=concentration
        )
        test_num_partitions, _ = distributions.shape

        # Assert
        for part in range(test_num_partitions):
            this_distribution = distributions[part]
            np.testing.assert_array_almost_equal(this_distribution, uniform, decimal=3)

    def test_create_dla_partitions_elements(self) -> None:
        """Test if partitions from Dirichlet Latent Allocation contain the same
        elements."""
        # Prepare
        num_partitions = 5
        concentration = 0.5

        # Execute
        partitions, _ = create_dla_partitions(
            dataset=self.ds, num_partitions=num_partitions, concentration=concentration
        )
        x_dla = np.concatenate([item[0] for item in partitions])
        y_dla = np.concatenate([item[1] for item in partitions])

        # Assert
        assert_identity(xy_0=self.ds, xy_1=(x_dla, y_dla))


if __name__ == "__main__":
    unittest.main(verbosity=2)
