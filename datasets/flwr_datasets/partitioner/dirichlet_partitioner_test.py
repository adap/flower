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
"""Test DirichletPartitioner."""
import unittest
from typing import Tuple, Union

import numpy as np
from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.dirichlet_partitioner import DirichletPartitioner


def _dummy_setup(
    num_partitions: int,
    alpha: Union[float, np.ndarray],
    num_rows: int,
    partition_by: str,
    self_balancing: bool = True,
) -> Tuple[Dataset, DirichletPartitioner]:
    """Create a dummy dataset and partitioner for testing."""
    data = {
        partition_by: [i % 3 for i in range(num_rows)],
        "features": list(range(num_rows)),
    }
    dataset = Dataset.from_dict(data)
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions,
        alpha=alpha,
        partition_by=partition_by,
        self_balancing=self_balancing,
    )
    partitioner.dataset = dataset
    return dataset, partitioner


class TestDirichletPartitioner(unittest.TestCase):
    """Test DirichletPartitioner."""

    @parameterized.expand(
        [
            # num_partitions, alpha, num_rows, partition_by
            (3, 0.5, 100, "labels"),
            (5, 1.0, 150, "labels"),
        ]
    )
    def test_valid_initialization(self, num_partitions, alpha, num_rows, partition_by):
        """Test if alpha is correct scaled based on the given num_partitions."""
        dataset, partitioner = _dummy_setup(
            num_partitions, alpha, num_rows, partition_by
        )
        self.assertEqual(
            (
                partitioner._num_partitions,
                len(partitioner._alpha),
                partitioner._partition_by,
            ),
            (num_partitions, num_partitions, partition_by),
        )

    def test_invalid_num_partitions(self):
        """Test if 0 is invalid num_partitions."""
        dataset, partitioner = _dummy_setup(
            num_partitions=0, alpha=1.0, num_rows=100, partition_by="labels"
        )
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)

    def test_invalid_alpha(self):
        """Test alpha list len not matching the num_partitions."""
        with self.assertRaises(ValueError):
            DirichletPartitioner(
                num_partitions=3, alpha=[0.5, 0.5], partition_by="labels"
            )

    def test_load_partition(self):
        _, partitioner = _dummy_setup(3, 0.5, 100, "labels")
        partition_list = [partitioner.load_partition(node_id) for node_id in [0,1,2]]
        self.assertGreaterEqual(all([len(p) for p in partition_list]), partitioner._min_partition_size)

    def test_load_invalid_partition_index(self):
        """Test if raises when the load_partition is above the num_partitions."""
        _, partitioner = _dummy_setup(3, 0.5, 100, "labels")
        with self.assertRaises(KeyError):
            partitioner.load_partition(3)

    def test_alpha_initialization(self):
        """Test alpha does not change when in NDArrayFloat format."""
        _, partitioner = _dummy_setup(3, np.array([1.0, 1.0, 1.0]), 100, "labels")
        self.assertTrue(np.all(partitioner._alpha == np.array([1.0, 1.0, 1.0])))

    def test__determine_node_id_to_indices(self):
        """Test the determine_nod_id_to_indices matches the flag after the call."""
        num_partitions, alpha, num_rows, partition_by = 3, 0.5, 100, "labels"
        _, partitioner = _dummy_setup(num_partitions, alpha, num_rows, partition_by)
        partitioner._determine_node_id_to_indices_if_needed()
        self.assertTrue(
            partitioner._node_id_to_indices_determined
            and len(partitioner._node_id_to_indices) == num_partitions
        )


# todo: write tests for the alpha values (make sure they are positive)

if __name__ == "__main__":
    unittest.main()
