"""Non_iid partitioner tests."""


import itertools
import unittest
from typing import Tuple

import numpy as np
from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.grouped_natural_id_partitioner import (
    GroupedNaturalIdPartitioner,
)


def _dummy_setup(
    num_rows: int, n_unique_natural_ids: int, num_nodes: int
) -> Tuple[Dataset, GroupedNaturalIdPartitioner]:
    """Create a dummy dataset and partitioner based on given arguments.

    The partitioner has automatically the dataset assigned to it.
    """
    dataset = _create_dataset(num_rows, n_unique_natural_ids)
    partitioner = GroupedNaturalIdPartitioner(
        partition_by="natural_id", num_groups=num_nodes
    )
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


class TestGroupedNaturalIdPartitioner(unittest.TestCase):
    """Test IidPartitioner."""

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5], [1, 3, 4, 4]))
    )
    def test_load_partition_num_partitions(
        self, num_rows: int, num_unique_natural_id: int, num_groups: int
    ) -> None:
        """Test if the number of partitions match the number of nodes."""
        _, partitioner = _dummy_setup(num_rows, num_unique_natural_id, num_groups)
        # Simulate usage to start lazy node_id_to_natural_id creation
        _ = partitioner.load_partition(0)
        self.assertEqual(
            len(partitioner.node_id_to_natural_id.keys()),
            num_groups,
            f"part: {partitioner._group_size}, real: {num_groups}",
        )

    @parameterized.expand(  # type: ignore
        # num_rows, num_unique_natural_ids
        list(itertools.product([10, 30, 100, 1000], [2, 3, 4, 5], [1, 3, 4, 4]))
    )
    def test_load_partition_correct_division(
        self, num_rows: int, num_unique_natural_ids: int, num_groups: int
    ) -> None:
        """Test if the data in partitions is correct.

        Only the correct data is tested in this method.
        """
        true_labels = np.array_split(
            np.array(range(num_unique_natural_ids)).astype(str), num_groups
        )

        _, partitioner = _dummy_setup(num_rows, num_unique_natural_ids, num_groups)
        dataset = partitioner.load_partition(0)
        labels = np.unique(dataset["natural_id"])

        self.assertTrue(
            (labels == true_labels[0]).all(),
            f"labels: {labels}; true_labels: {true_labels[0]}",
        )


if __name__ == "__main__":
    unittest.main()
