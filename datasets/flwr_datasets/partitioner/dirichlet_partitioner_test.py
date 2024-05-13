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
"""Test DirichletPartitioner."""


# pylint: disable=W0212
import unittest
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.dirichlet_partitioner import DirichletPartitioner


def _dummy_setup(
    num_partitions: int,
    alpha: Union[float, NDArray[np.float_]],
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


class TestDirichletPartitionerSuccess(unittest.TestCase):
    """Test DirichletPartitioner used with no exceptions."""

    @parameterized.expand(  # type: ignore
        [
            # num_partitions, alpha, num_rows, partition_by
            (3, 0.5, 100, "labels"),
            (5, 1.0, 150, "labels"),
        ]
    )
    def test_valid_initialization(
        self, num_partitions: int, alpha: float, num_rows: int, partition_by: str
    ) -> None:
        """Test if alpha is correct scaled based on the given num_partitions."""
        _, partitioner = _dummy_setup(num_partitions, alpha, num_rows, partition_by)
        self.assertEqual(
            (
                partitioner._num_partitions,
                len(partitioner._alpha),
                partitioner._partition_by,
            ),
            (num_partitions, num_partitions, partition_by),
        )

    def test_min_partition_size_requirement(self) -> None:
        """Test if partitions are created with min partition size required."""
        _, partitioner = _dummy_setup(3, 0.5, 100, "labels")
        partition_list = [
            partitioner.load_partition(partition_id) for partition_id in [0, 1, 2]
        ]
        self.assertTrue(
            all(len(p) > partitioner._min_partition_size for p in partition_list)
        )

    def test_alpha_in_ndarray_initialization(self) -> None:
        """Test alpha does not change when in NDArrayFloat format."""
        _, partitioner = _dummy_setup(3, np.array([1.0, 1.0, 1.0]), 100, "labels")
        self.assertTrue(np.all(partitioner._alpha == np.array([1.0, 1.0, 1.0])))

    def test__determine_partition_id_to_indices(self) -> None:
        """Test the determine_nod_id_to_indices matches the flag after the call."""
        num_partitions, alpha, num_rows, partition_by = 3, 0.5, 100, "labels"
        _, partitioner = _dummy_setup(num_partitions, alpha, num_rows, partition_by)
        partitioner._determine_partition_id_to_indices_if_needed()
        self.assertTrue(
            partitioner._partition_id_to_indices_determined
            and len(partitioner._partition_id_to_indices) == num_partitions
        )


class TestDirichletPartitionerFailure(unittest.TestCase):
    """Test DirichletPartitioner failures (exceptions) by incorrect usage."""

    @parameterized.expand([(-2,), (-1,), (3,), (4,), (100,)])  # type: ignore
    def test_load_invalid_partition_index(self, partition_id):
        """Test if raises when the load_partition is above the num_partitions."""
        _, partitioner = _dummy_setup(3, 0.5, 100, "labels")
        with self.assertRaises(KeyError):
            partitioner.load_partition(partition_id)

    @parameterized.expand(  # type: ignore
        [
            # alpha, num_partitions
            (-0.5, 1),
            (-0.5, 2),
            (-0.5, 3),
            (-0.5, 10),
            ([0.5, 0.5, -0.5], 3),
            ([-0.5, 0.5, -0.5], 3),
            ([-0.5, 0.5, 0.5], 3),
            ([-0.5, -0.5, -0.5], 3),
            ([0.5, 0.5, -0.5, -0.5, 0.5], 5),
            (np.array([0.5, 0.5, -0.5]), 3),
            (np.array([-0.5, 0.5, -0.5]), 3),
            (np.array([-0.5, 0.5, 0.5]), 3),
            (np.array([-0.5, -0.5, -0.5]), 3),
            (np.array([0.5, 0.5, -0.5, -0.5, 0.5]), 5),
        ]
    )
    def test_negative_values_in_alpha(self, alpha, num_partitions):
        """Test if giving the negative value of alpha raises error."""
        num_rows, partition_by = 100, "labels"
        with self.assertRaises(ValueError):
            _, _ = _dummy_setup(num_partitions, alpha, num_rows, partition_by)

    @parameterized.expand(  # type: ignore
        [
            # alpha, num_partitions
            # alpha greater than the num_partitions
            ([0.5, 0.5], 1),
            ([0.5, 0.5, 0.5], 2),
            (np.array([0.5, 0.5]), 1),
            (np.array([0.5, 0.5, 0.5]), 2),
            (np.array([0.5, 0.5, 0.5, 0.5]), 3),
        ]
    )
    def test_incorrect_alpha_shape(self, alpha, num_partitions):
        """Test alpha list len not matching the num_partitions."""
        with self.assertRaises(ValueError):
            DirichletPartitioner(
                num_partitions=num_partitions, alpha=alpha, partition_by="labels"
            )

    @parameterized.expand(  # type: ignore
        [(0,), (-1,), (11,), (100,)]
    )  # num_partitions,
    def test_invalid_num_partitions(self, num_partitions):
        """Test if 0 is invalid num_partitions."""
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(
                num_partitions=num_partitions,
                alpha=1.0,
                num_rows=10,
                partition_by="labels",
            )
            partitioner.load_partition(0)


if __name__ == "__main__":
    unittest.main()
