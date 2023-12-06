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
            (3, 0.5, 100, "labels"),
            (5, 1.0, 150, "labels"),
        ]
    )
    def test_valid_initialization(self, num_partitions, alpha, num_rows, partition_by):
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
        dataset, partitioner = _dummy_setup(
            num_partitions=0, alpha=1.0, num_rows=100, partition_by="labels"
        )
        with self.assertRaises(ValueError):
            partitioner.load_partition(0)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            DirichletPartitioner(
                num_partitions=3, alpha=[0.5, 0.5], partition_by="labels"
            )

    @parameterized.expand(
        [
            (3, 0.5, 100, "labels", 0),
            (3, 0.5, 100, "labels", 2),
        ]
    )
    def test_load_partition(
        self, num_partitions, alpha, num_rows, partition_by, node_id
    ):
        _, partitioner = _dummy_setup(num_partitions, alpha, num_rows, partition_by)
        partition = partitioner.load_partition(node_id)
        self.assertGreaterEqual(len(partition), partitioner._min_partition_size)

    def test_load_invalid_partition_index(self):
        _, partitioner = _dummy_setup(3, 0.5, 100, "labels")
        with self.assertRaises(KeyError):
            partitioner.load_partition(3)

    def test_alpha_initialization(self):
        _, partitioner = _dummy_setup(3, np.array([1.0, 1.0, 1.0]), 100, "labels")
        self.assertTrue(np.all(partitioner._alpha == np.array([1.0, 1.0, 1.0])))

    def test_internal_method_determine_node_id_to_indices(self):
        num_partitions, alpha, num_rows, partition_by = 3, 0.5, 100, "labels"
        _, partitioner = _dummy_setup(num_partitions, alpha, num_rows, partition_by)
        partitioner._determine_node_id_to_indices_if_needed()
        self.assertTrue(
            partitioner._node_id_to_indices_determined
            and len(partitioner._node_id_to_indices) == num_partitions
        )

    def test_edge_case_small_dataset(self):
        num_partitions, alpha, num_rows, partition_by = 3, 0.5, 3, "labels"
        _, partitioner = _dummy_setup(num_partitions, alpha, num_rows, partition_by)
        partitioner._determine_node_id_to_indices_if_needed()
        min_partition_size = all(
            len(indices) >= partitioner._min_partition_size
            for indices in partitioner._node_id_to_indices.values()
        )
        self.assertTrue(min_partition_size)
# todo: write tests for the alpha values (make sure they are positive)

if __name__ == "__main__":
    unittest.main()
