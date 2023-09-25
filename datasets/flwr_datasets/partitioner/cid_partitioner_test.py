# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable alaw or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CID partitioner tests."""
import itertools
import math
import unittest
from typing import Tuple

from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.cid_partitioner import CidPartitioner


def _dummy_setup(num_rows: int, n_unique_cids: int) -> Tuple[Dataset, CidPartitioner]:
    """Create a dummy dataset and partitioner based on given arguments.

    The partitioner has automatically the dataset assigned to it.
    """
    dataset = _create_dataset(num_rows, n_unique_cids)
    partitioner = CidPartitioner(partition_by="cid")
    partitioner.dataset = dataset
    return dataset, partitioner


def _create_dataset(num_rows: int, n_unique_cids: int):
    """Create dataset based on the number of rows and unique cids."""
    data = {
        "features": list(range(num_rows)),
        "cid": [f"{i % n_unique_cids}" for i in range(num_rows)],
        "labels": [i % 2 for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    return dataset


class TestCidPartitioner(unittest.TestCase):
    """Test IidPartitioner."""

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_cids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5]))
    )
    def test_load_partition_num_partitions(
        self, num_rows: int, num_unique_cid: int
    ) -> None:
        """Test if the number of partitions match the number of unique cids.

        Only the correct data is tested in this method.
        """
        _, partitioner = _dummy_setup(num_rows, num_unique_cid)
        # Simulate usage to start lazy index_to_cid creation
        _ = partitioner.load_partition(0)
        self.assertEqual(len(partitioner.index_to_cid), num_unique_cid)

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_cids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5]))
    )
    def test_load_partition_max_partition_size(
        self, num_rows: int, num_unique_cid: int
    ) -> None:
        """Test if the number of partitions match the number of unique cids.

        Only the correct data is tested in this method.
        """
        print(num_rows)
        print(num_unique_cid)
        _, partitioner = _dummy_setup(num_rows, num_unique_cid)
        max_size = max(
            [len(partitioner.load_partition(i)) for i in range(num_unique_cid)]
        )
        self.assertEqual(max_size, math.ceil(num_rows / num_unique_cid))

    def test_partitioner_with_non_existing_column_partition_by(self) -> None:
        """Test error when the partition_by columns does not exist."""
        dataset = _create_dataset(10, 2)
        partitioner = CidPartitioner(partition_by="not-cid")
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)

    # @parameterized.expand(  # type: ignore
    #     [
    #         # num_partitions, num_rows, partition_index
    #         (10, 10, 10),
    #         (10, 10, -1),
    #         (10, 10, 11),
    #         (10, 100, 1000),
    #         (5, 50, 60),
    #         (20, 200, 210),
    #     ]
    # )
    # def test_load_invalid_partition_index(
    #         self, num_partitions: int, num_rows: int, partition_index: int
    # ) -> None:
    #     """Test loading a partition with an index out of range."""
    #     _, partitioner = _dummy_setup(num_partitions, num_rows)
    #     with self.assertRaises(ValueError):
    #         partitioner.load_partition(partition_index)
    #
    # def test_is_dataset_assigned_false(self) -> None:
    #     """Test if the is_dataset_assigned method works correctly if not assigned."""
    #     partitioner = IidPartitioner(num_partitions=10)
    #
    #     # Initially, the dataset should not be assigned
    #     self.assertFalse(partitioner.is_dataset_assigned())
    #
    # def test_is_dataset_assigned_true(self) -> None:
    #     """Test if the is_dataset_assigned method works correctly if assigned."""
    #     num_partitions = 10
    #     num_rows = 100
    #     _, partitioner = _dummy_setup(num_partitions, num_rows)
    #     self.assertTrue(partitioner.is_dataset_assigned())
    #
    # def test_dataset_setter(self) -> None:
    #     """Test the dataset.setter method."""
    #     num_partitions = 10
    #     num_rows = 100
    #     dataset, partitioner = _dummy_setup(num_partitions, num_rows)
    #
    #     # It should not allow setting the dataset a second time
    #     with self.assertRaises(Exception) as context:
    #         partitioner.dataset = dataset
    #     self.assertIn(
    #         "The dataset should be assigned only once", str(context.exception)
    #     )
    #
    # def test_dataset_getter_raises(self) -> None:
    #     """Test the dataset getter method."""
    #     num_partitions = 10
    #     partitioner = IidPartitioner(num_partitions=num_partitions)
    #     with self.assertRaises(AttributeError) as context:
    #         _ = partitioner.dataset
    #     self.assertIn(
    #         "The dataset field should be set before using it", str(context.exception)
    #     )
    #
    # def test_dataset_getter_used_correctly(self) -> None:
    #     """Test the dataset getter method."""
    #     num_partitions = 10
    #     num_rows = 100
    #     dataset, partitioner = _dummy_setup(num_partitions, num_rows)
    #     # After setting, it should return the dataset
    #     self.assertEqual(partitioner.dataset, dataset)


if __name__ == "__main__":
    unittest.main()
