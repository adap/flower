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
"""Test ContinuousPartitioner."""

# pylint: disable=W0212, R0801, R0917, R0913
import unittest
from typing import Optional

import numpy as np
from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.continuous_partitioner import ContinuousPartitioner


def _dummy_setup(
    num_partitions: int,
    strictness: float,
    num_rows: int,
    partition_by: str = "score",
    shuffle: bool = True,
    seed: Optional[int] = 42,
) -> tuple[Dataset, ContinuousPartitioner]:
    """Create a dummy dataset and partitioner for testing."""
    scores = np.linspace(0, 10, num_rows)
    data = {
        partition_by: scores.tolist(),
        "features": np.random.randn(num_rows).tolist(),
    }
    dataset = Dataset.from_dict(data)
    partitioner = ContinuousPartitioner(
        num_partitions=num_partitions,
        partition_by=partition_by,
        strictness=strictness,
        shuffle=shuffle,
        seed=seed,
    )
    partitioner.dataset = dataset
    return dataset, partitioner


class TestContinuousPartitionerSuccess(unittest.TestCase):
    """Test ContinuousPartitioner used with no exceptions."""

    @parameterized.expand(
        [
            (5, 0.0, 100),
            (5, 1.0, 100),
            (3, 0.5, 50),
        ]
    )  # type: ignore
    def test_valid_partition_shapes(
        self, num_partitions: int, strictness: float, num_rows: int
    ) -> None:
        """Check each partition has non-zero and collectively complete indices."""
        _, partitioner = _dummy_setup(num_partitions, strictness, num_rows)
        partition_sizes = [
            len(partitioner.load_partition(pid)) for pid in range(num_partitions)
        ]
        self.assertEqual(sum(partition_sizes), num_rows)
        self.assertTrue(all(size > 0 for size in partition_sizes))

    def test_partition_determinism_with_fixed_seed(self) -> None:
        """Check partitioning is deterministic given fixed seed and shuffle=True."""
        _, partitioner_1 = _dummy_setup(4, 0.7, 40, seed=123)
        _, partitioner_2 = _dummy_setup(4, 0.7, 40, seed=123)
        idx1 = partitioner_1.partition_id_to_indices
        idx2 = partitioner_2.partition_id_to_indices
        self.assertEqual(idx1, idx2)

    def test_partitioning_different_strictness(self) -> None:
        """Check different strictness lead to different groupings."""
        _, p_lo = _dummy_setup(4, 0.1, 40, seed=42)
        _, p_hi = _dummy_setup(4, 0.9, 40, seed=42)
        self.assertNotEqual(p_lo.partition_id_to_indices, p_hi.partition_id_to_indices)

    def test_num_partitions_property(self) -> None:
        """Test num_partitions property returns correct value."""
        _, partitioner = _dummy_setup(3, 0.5, 30)
        self.assertEqual(partitioner.num_partitions, 3)

    def test_shuffle_false_gives_sorted_groups(self) -> None:
        """Test when shuffle=False, partition indices are sorted."""
        _, partitioner = _dummy_setup(3, 1.0, 30, shuffle=False)
        for indices in partitioner.partition_id_to_indices.values():
            self.assertTrue(indices == sorted(indices))

    def test_strictness_zero_is_pure_noise(self) -> None:
        """With strictness=0, output should be similar to shuffling."""
        _, part1 = _dummy_setup(3, strictness=0.0, num_rows=30, seed=42)
        _, part2 = _dummy_setup(3, strictness=0.0, num_rows=30, seed=43)
        self.assertNotEqual(
            part1.partition_id_to_indices, part2.partition_id_to_indices
        )

    def test_partitions_are_disjoint(self) -> None:
        """Ensure no sample index is shared across partitions."""
        _, partitioner = _dummy_setup(5, 0.8, 50)
        all_indices = sum(partitioner.partition_id_to_indices.values(), [])
        self.assertEqual(len(all_indices), len(set(all_indices)))

    def test_partition_per_sample(self) -> None:
        """Every row in its own partition (extreme non-iid case)."""
        _, partitioner = _dummy_setup(10, 1.0, 10)
        for i in range(10):
            self.assertEqual(len(partitioner.load_partition(i)), 1)

    def test_monotonic_increasing_across_partitions_when_sorted(self) -> None:
        """With no shuffling and max strictness, partitions follow increasing order."""
        _, partitioner = _dummy_setup(3, 1.0, 30, shuffle=False)
        values: list[float] = []
        for pid in range(3):
            indices = partitioner.partition_id_to_indices[pid]
            values.extend(partitioner.dataset["score"][i] for i in indices)
        self.assertEqual(values, sorted(values))


class TestContinuousPartitionerFailure(unittest.TestCase):
    """Test ContinuousPartitioner failures by incorrect usage."""

    def test_invalid_num_partitions(self) -> None:
        """Raise ValueError for non-positive num_partitions."""
        with self.assertRaises(ValueError):
            _ = _dummy_setup(0, 0.5, 10)

    def test_invalid_strictness(self) -> None:
        """Raise ValueError for strictness outside [0, 1]."""
        with self.assertRaises(ValueError):
            _ = _dummy_setup(3, -0.1, 10)
        with self.assertRaises(ValueError):
            _ = _dummy_setup(3, 1.1, 10)

    def test_missing_column_raises(self) -> None:
        """Raise KeyError when partition_by column is missing."""
        data = {
            "features": list(range(10)),
        }
        dataset = Dataset.from_dict(data)
        partitioner = ContinuousPartitioner(3, partition_by="missing", strictness=0.5)
        partitioner.dataset = dataset
        with self.assertRaises(KeyError):
            _ = partitioner.load_partition(0)

    def test_nan_value_in_column_raises(self) -> None:
        """Raise ValueError when partition_by column contains NaN or None."""
        data = {
            "score": [1.0, 2.0, None, 4.0, 5.0],
            "features": [0, 1, 2, 3, 4],
        }
        dataset = Dataset.from_dict(data)
        partitioner = ContinuousPartitioner(2, partition_by="score", strictness=0.5)
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            _ = partitioner.load_partition(0)

    def test_zero_stddev_raises(self) -> None:
        """Raise ValueError when all partition_by values are constant."""
        data = {
            "score": [3.14] * 10,  # Constant value
            "features": list(range(10)),
        }
        dataset = Dataset.from_dict(data)
        partitioner = ContinuousPartitioner(3, partition_by="score", strictness=1.0)
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            _ = partitioner.load_partition(0)

    def test_partition_id_out_of_range(self) -> None:
        """Raise KeyError if accessing a non-existent partition."""
        _, partitioner = _dummy_setup(2, 0.5, 10)
        with self.assertRaises(KeyError):
            _ = partitioner.load_partition(10)


if __name__ == "__main__":
    unittest.main()
