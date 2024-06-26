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
"""Test cases for ClassConstrainedPartitioner."""


import unittest
from typing import Dict

import numpy as np
from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.class_constrained_partitioner import (
    ClassConstrainedPartitioner,
)


def _dummy_dataset_setup(
    num_samples: int, partition_by: str, num_classes: int
) -> Dataset:
    """Create a dummy dataset for testing."""
    data = {
        partition_by: np.tile(np.arange(num_classes), num_samples // num_classes + 1)[
            :num_samples
        ],
        "features": np.random.randn(num_samples),
    }
    return Dataset.from_dict(data)


class TestClassConstrainedPartitioner(unittest.TestCase):
    """Unit tests for ClassConstrainedPartitioner."""

    @parameterized.expand(  # type: ignore
        [
            # num_partition, num_classes_per_partition, num_samples, total_classes
            (3, 1, 60, 3),  # Single class per partition scenario
            (5, 2, 100, 5),
            (5, 2, 100, 10),
            (4, 3, 120, 6),
        ]
    )
    def test_variable_class_partitioning(
        self,
        num_partitions: int,
        num_classes_per_partition: int,
        num_samples: int,
        total_classes: int,
    ) -> None:
        """Test correct number of unique classes."""
        dataset = _dummy_dataset_setup(num_samples, "labels", total_classes)
        partitioner = ClassConstrainedPartitioner(
            num_partitions=num_partitions,
            partition_by="labels",
            num_classes_per_partition=num_classes_per_partition,
        )
        partitioner.dataset = dataset
        partitions: Dict[int, Dataset] = {
            pid: partitioner.load_partition(pid) for pid in range(num_partitions)
        }
        unique_classes_per_partition = {
            pid: np.unique(partition["labels"]) for pid, partition in partitions.items()
        }

        for classes in unique_classes_per_partition.values():
            self.assertEqual(len(classes), num_classes_per_partition)

    def test_first_class_deterministic_assignment(self) -> None:
        """Test deterministic assignment of first classes to partitions."""
        dataset = _dummy_dataset_setup(100, "labels", 10)
        partitioner = ClassConstrainedPartitioner(
            num_partitions=10,
            partition_by="labels",
            num_classes_per_partition=2,
            first_class_deterministic_assignment=True,
        )
        partitioner.dataset = dataset
        partitioner.load_partition(0)
        # Expecting each class from 0 to 9 at least once in some partition
        expected_classes = set(range(10))
        actual_classes = set()
        for pid in range(10):
            partition = partitioner.load_partition(pid)
            actual_classes.update(np.unique(partition["labels"]))
        self.assertEqual(expected_classes, actual_classes)

    @parameterized.expand(  # type: ignore
        [
            # num_partitions, num_classes_per_partition, num_samples
            (10, 10, 5),  # More partitions than samples
            (10, 20, 100),  # More classes per partition than available classes (5)
        ]
    )
    def test_excessive_partitions(
        self, num_partitions: int, num_classes_per_partition: int, num_samples: int
    ) -> None:
        """Test edge cases with excessive partitions or classes per partition."""
        dataset = _dummy_dataset_setup(num_samples, "labels", 5)  # Only 5 classes
        with self.assertRaises(ValueError):
            partitioner = ClassConstrainedPartitioner(
                num_partitions=num_partitions,
                partition_by="labels",
                num_classes_per_partition=num_classes_per_partition,
            )
            partitioner.dataset = dataset
            partitioner.load_partition(0)


if __name__ == "__main__":
    unittest.main()
