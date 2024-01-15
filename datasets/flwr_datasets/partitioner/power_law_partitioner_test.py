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
"""Test PowerLawPartitioner."""
# pylint: disable=W0212, R0913
import unittest
from typing import Optional, Tuple

import numpy as np

from datasets import Dataset
from flwr_datasets.partitioner.power_law_partitioner import PowerLawPartitioner


def _dummy_setup(
    num_rows: int,
    num_partitions: int,
    partition_by: str,
    num_labels_per_partition: int,
    mean: float = 0.0,
    sigma: float = 2.0,
    min_partition_size: Optional[int] = None,
    n_classes_to_preassign: int = 2,
    n_samples_per_class_to_preassign: int = 5,
) -> Tuple[Dataset, PowerLawPartitioner]:
    """Create a dummy dataset and partitioner for testing."""
    data = {
        partition_by: [i % 3 for i in range(num_rows)],
        "features": list(range(num_rows)),
    }
    dataset = Dataset.from_dict(data)
    partitioner = PowerLawPartitioner(
        num_partitions=num_partitions,
        partition_by=partition_by,
        num_labels_per_partition=num_labels_per_partition,
        mean=mean,
        sigma=sigma,
        min_partition_size=min_partition_size,
        n_classes_to_preassign=n_classes_to_preassign,
        n_samples_per_class_to_preassign=n_samples_per_class_to_preassign,
    )
    partitioner.dataset = dataset
    return dataset, partitioner


class TestPowerLawPartitioner(unittest.TestCase):
    """Test PowerLawPartitioner."""

    def test_correct_num_of_partitions(self) -> None:
        """Test correct number of partitions."""
        num_rows = 113
        partition_by = "label"
        num_partitions = 3
        num_labels_per_partition = 2
        mean = 0.0
        sigma = 2.0
        min_partition_size = 0
        n_classes_to_preassign = 2
        n_samples_per_class_to_preassign = 5

        _, partitioner = _dummy_setup(
            num_rows,
            num_partitions,
            partition_by,
            num_labels_per_partition,
            mean,
            sigma,
            min_partition_size,
            n_classes_to_preassign,
            n_samples_per_class_to_preassign,
        )
        _ = partitioner.load_partition(0)
        self.assertEqual(len(partitioner._node_id_to_indices.keys()), num_partitions)

    def test_correct_number_of_unique_labels(self) -> None:
        """Test correct number of unique labels."""
        num_rows = 113
        partition_by = "label"
        num_partitions = 3
        num_labels_per_partition = 2
        mean = 0.0
        sigma = 2.0
        min_partition_size = 0
        n_classes_to_preassign = 2
        n_samples_per_class_to_preassign = 5

        _, partitioner = _dummy_setup(
            num_rows,
            num_partitions,
            partition_by,
            num_labels_per_partition,
            mean,
            sigma,
            min_partition_size,
            n_classes_to_preassign,
            n_samples_per_class_to_preassign,
        )
        num_unique_labels = np.array(
            [
                len(np.unique(partitioner.load_partition(i)[partition_by]))
                for i in range(num_partitions)
            ]
        )
        self.assertTrue((num_unique_labels == num_labels_per_partition).all())


if __name__ == "__main__":
    unittest.main()
