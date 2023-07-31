from datasets import Dataset
import unittest

from sampler import IIDSampler, CIDSampler


class TestIIDSampler(unittest.TestCase):
    """Test IIDSampler."""

    def setUp(self):
        """Create a dummy dataset with 100 rows, numerical features, and labels."""
        data = {
            "features": [i for i in range(100)],
            "labels": [i % 2 for i in range(100)]
        }
        self.dataset = Dataset.from_dict(data)
        self.n_partitions = 10
        self.partition_size = self.dataset.num_rows // self.n_partitions
        self.sampler = IIDSampler(n_partitions=self.n_partitions)

    def test_get_partition(self):
        partition_index = 2
        partition = self.sampler.get_partition(self.dataset, partition_index)
        self.assertEqual(partition.num_rows, self.partition_size)
        self.assertEqual(partition["features"][0],
                         partition_index * self.partition_size)

    def test_get_partitions(self):
        partitions = self.sampler.get_partitions(self.dataset)
        for i, partition in enumerate(partitions):
            self.assertEqual(partition.num_rows, self.partition_size)
            self.assertEqual(partition["features"][0], i * self.partition_size)

    def test_get_partitions_size(self):
        partitions = self.sampler.get_partitions(self.dataset)
        self.assertEqual(len(partitions), self.n_partitions)


class TestCIDSampler(unittest.TestCase):
    """Test CIDSampler."""

    def setUp(self):
        """Create a 100-row dummy dataset with numerical features, labels, and cids."""
        data = {
            "features": [i for i in range(100)],
            "labels": [i % 2 for i in range(100)],
            "cid": [str(i // 10) for i in range(100)]  # 10 different cids
        }
        self.dataset = Dataset.from_dict(data)
        self.n_partitions = 10
        self.partition_size = self.dataset.num_rows // self.n_partitions
        self.partition_by = "cid"
        self.sampler = CIDSampler(n_partitions=self.n_partitions,
                                  partition_by=self.partition_by)

    def test_create_int_idx_to_cid_size(self):
        self.sampler._create_int_idx_to_cid(self.dataset)
        self.assertEqual(len(self.sampler._index_to_cid), self.n_partitions)

    def test_create_int_idx_to_cid_mapping(self):
        self.sampler._create_int_idx_to_cid(self.dataset)
        self.assertEqual(int(self.sampler._index_to_cid[0]), 0)

    def test_get_partition(self):
        partition_index = 2
        partition = self.sampler.get_partition(self.dataset, partition_index)
        self.assertEqual(partition["cid"][0],
                         self.sampler._index_to_cid[partition_index])

    def test_partitions_have_single_unique_cid(self):
        partitions = self.sampler.get_partitions(self.dataset)
        for partition in partitions:
            unique_cids = set(partition['cid'])
            self.assertEqual(len(unique_cids), 1)


if __name__ == "__main__":
    unittest.main()
