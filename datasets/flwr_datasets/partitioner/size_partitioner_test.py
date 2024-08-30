# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Test the SizePartitioner class."""

# pylint: disable=W0212
import unittest
from typing import Sequence

from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.size_partitioner import SizePartitioner


def _dummy_setup_size(partition_sizes: Sequence[int], num_rows: int) -> SizePartitioner:
    """Create a dummy dataset and SizePartitioner for testing."""
    data = {
        "features": list(range(num_rows)),
    }
    dataset = Dataset.from_dict(data)
    partitioner = SizePartitioner(partition_sizes=partition_sizes)
    partitioner.dataset = dataset
    return partitioner


tested_valid_intits = [
    ((10, 20, 30), 60),
    # Non growing order
    ((20, 40, 10), 70),
    # Different lengths
    ((10, 10), 20),
    # Single partition
    ((10,), 10),
]


class TestSizePartitionerSuccess(unittest.TestCase):
    """Test SizePartitioner used with no exceptions."""

    @parameterized.expand(tested_valid_intits)  # type: ignore
    def test_valid_initialization(
        self, partition_sizes: Sequence[int], dataset_size: int
    ) -> None:
        """Test that the SizePartitioner initializes correctly with valid sizes."""
        partitioner = _dummy_setup_size(partition_sizes, dataset_size)
        self.assertEqual(partitioner.num_partitions, len(partition_sizes))

    @parameterized.expand(tested_valid_intits)  # type: ignore
    def test_partition_size_assignment(
        self, partition_sizes: Sequence[int], dataset_size: int
    ) -> None:
        """Test that partitions are assigned the correct size."""
        partitioner = _dummy_setup_size(partition_sizes, dataset_size)
        partitioner._determine_partition_id_to_indices_if_needed()
        self.assertEqual(
            {
                pid: len(indices)
                for pid, indices in partitioner.partition_id_to_indices.items()
            },
            dict(enumerate(partition_sizes)),
        )

    def test_correct_partition_loading(self) -> None:
        """Test that partitions are loaded correctly."""
        partition_sizes = [10, 20, 30]
        partitioner = _dummy_setup_size(partition_sizes, 60)
        partition = partitioner.load_partition(1)
        self.assertEqual(len(partition), 20)

    def test_warning_for_smaller_partition_sizes(self) -> None:
        """Test a warning is raised if sum of partition sizes < len(ds)."""
        partition_sizes = [10, 5, 20]
        partitioner = _dummy_setup_size(partition_sizes, 50)
        with self.assertWarns(Warning):
            partitioner._determine_partition_id_to_indices_if_needed()

    def test_no_exception_for_exact_size(self) -> None:
        """Test no exception is raised when len(ds) == sum(patition_sizes)."""
        partition_sizes = [10, 20, 30]
        partitioner = _dummy_setup_size(partition_sizes, 60)
        partitioner._determine_partition_id_to_indices_if_needed()


class TestSizePartitionerFailure(unittest.TestCase):
    """Test SizePartitioner failures (exceptions) by incorrect usage."""

    def test_invalid_partition_size(self) -> None:
        """Test if raises ValueError when partition sizes are non-positive."""
        with self.assertRaises(ValueError):
            SizePartitioner(partition_sizes=[-1, 10, 20])

    def test_invalid_partition_type(self) -> None:
        """Test if raises ValueError when partition sizes are non-positive."""
        with self.assertRaises(ValueError):
            SizePartitioner(partition_sizes=[0.2, 0.3])  # type: ignore[list-item]

    def test_partition_size_exceeds_dataset(self) -> None:
        """Test if raises ValueError when partition sizes exceed dataset size."""
        partition_sizes = [10, 20, 30]
        partitioner = _dummy_setup_size(partition_sizes, 40)
        with self.assertRaises(ValueError):
            partitioner._determine_partition_id_to_indices_if_needed()

    def test_load_invalid_partition_index(self) -> None:
        """Test if raises KeyError when an invalid partition index is loaded."""
        partition_sizes = [10, 20, 30]
        partitioner = _dummy_setup_size(partition_sizes, 60)
        with self.assertRaises(KeyError):
            partitioner.load_partition(3)


if __name__ == "__main__":
    unittest.main()
