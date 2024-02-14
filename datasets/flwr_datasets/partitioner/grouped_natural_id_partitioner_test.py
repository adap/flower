"""Non_iid partitioner tests."""


import itertools
import unittest
from typing import Tuple
import numpy as np

from parameterized import parameterized

from datasets import Dataset
from datasets.flwr_datasets.partitioner.grouped_natural_id_partitioner import GroupedNaturalIdPartitioner

def _dummy_setup(
    num_rows: int, n_unique_natural_ids: int, num_nodes: int
) -> Tuple[Dataset, GroupedNaturalIdPartitioner]:
    """Create a dummy dataset and partitioner based on given arguments.

    The partitioner has automatically the dataset assigned to it.
    """
    dataset = _create_dataset(num_rows, n_unique_natural_ids)
    partitioner = GroupedNaturalIdPartitioner(partition_by="natural_id",num_nodes=num_nodes)
    partitioner.dataset = dataset
    return dataset, partitioner


def _create_dataset(num_rows: int, n_unique_natural_ids: int) -> Dataset:
    """Create dataset based on the number of rows and unique natural ids."""

    data = {
        "features": list(range(num_rows)),
        "natural_id": [f"{i % n_unique_natural_ids}" for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    return dataset


class TestNaturalIdPartitioner(unittest.TestCase):
    """Test IidPartitioner."""

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5], [1,3,4,4]))
    )
    def test_load_partition_num_partitions(
        self, num_rows: int, num_unique_natural_id: int, num_nodes: int
    ) -> None:
        """Test if the number of partitions match the number of nodes.

        """
        _, partitioner = _dummy_setup(num_rows, num_unique_natural_id, num_nodes)
        # Simulate usage to start lazy node_id_to_natural_id creation
        _ = partitioner.load_partition(0)
        self.assertEqual(len(partitioner.node_id_to_natural_id.keys()), num_nodes, f'part: {partitioner._num_nodes}, real: {num_nodes}')

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5], [1,3,4,4]))
    )
    def test_load_partition_correct_division(
        self, num_rows: int, num_unique_natural_ids: int, num_nodes: int
    ) -> None:
        """Test if the data in partitions is correct.

        Only the correct data is tested in this method.
        """

        true_labels = np.array_split(np.array(range(num_unique_natural_ids)).astype(str), num_nodes)

        _, partitioner = _dummy_setup(num_rows, num_unique_natural_ids, num_nodes)
        dataset = partitioner.load_partition(0)
        labels = np.unique(dataset['natural_id'])

        self.assertTrue((labels == true_labels[0]).all(), f'labels: {labels}; true_labels: {true_labels[0]}')

    def test_partitioner_with_non_existing_column_partition_by(self) -> None:
        """Test error when the partition_by columns does not exist."""
        dataset = _create_dataset(10, 2)
        partitioner = GroupedNaturalIdPartitioner(partition_by="not-existing",num_nodes=2)
        partitioner.dataset = dataset
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)



    def test_cannot_set_node_id_to_natural_id(self) -> None:
        """Test the lack of ability to set node_id_to_natural_id."""
        _, partitioner = _dummy_setup(num_rows=10, n_unique_natural_ids=2, num_nodes=3)
        with self.assertRaises(AttributeError):
            partitioner.node_id_to_natural_id = {0: "0"}



if __name__ == "__main__":
    unittest.main()