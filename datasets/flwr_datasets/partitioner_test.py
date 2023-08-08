"""Partitioner tests."""
import unittest

from partitioner import IidPartitioner

from datasets import Dataset


class TestIidPartitioner(unittest.TestCase):
    """Test IidPartitioner."""

    def setUp(self) -> None:
        """Create a dummy dataset with 100 rows, numerical features, and labels."""
        data = {
            "features": list(range(100)),
            "labels": [i % 2 for i in range(100)],
        }
        self.dataset = Dataset.from_dict(data)
        self.num_partitions = 10
        self.partition_size = self.dataset.num_rows // self.num_partitions
        self.sampler = IidPartitioner(num_partitions=self.num_partitions)
        self.sampler.dataset = self.dataset

    def test_load_partition_size(self) -> None:
        """Test if the partition size matches the manually computed size."""
        partition_index = 2
        partition = self.sampler.load_partition(partition_index)
        self.assertEqual(len(partition), self.partition_size)

    def test_load_partition_correct_data(self) -> None:
        """Test if the data in partition is equal to the expected."""
        partition_index = 2
        partition = self.sampler.load_partition(partition_index)
        self.assertEqual(
            partition["features"][0], partition_index * self.partition_size
        )


if __name__ == "__main__":
    unittest.main()
