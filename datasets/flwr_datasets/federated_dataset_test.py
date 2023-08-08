"""Federated Dataset tests."""
import unittest

import pytest
from federated_dataset import FederatedDataset
from parameterized import parameterized, parameterized_class

import datasets


@parameterized_class(
    [
        {"dataset_name": "mnist"},
        {"dataset_name": "cifar10"},
    ]
)
class RealDatasetsFederatedDatasets(unittest.TestCase):
    """Test Real Dataset (MNIST, CIFAR10) in FederatedDatasets."""

    dataset_name = ""

    @parameterized.expand(
        [
            (
                "10",
                10,
            ),
            (
                "100",
                100,
            ),
        ]
    )
    def test_load_partition_size(self, _: str, train_num_partitions: int):
        """Test if the partition size is correct based on the number of partitions."""
        dataset_fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": train_num_partitions}
        )
        dataset_partition0 = dataset_fds.load_partition(0, "train")
        dataset = datasets.load_dataset(self.dataset_name)
        self.assertEqual(
            len(dataset_partition0), len(dataset["train"]) // train_num_partitions
        )

    def test_load_full(self):
        """Test if the load_full works on the correct split name."""
        dataset_fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": 100}
        )
        dataset_fds_test = dataset_fds.load_full("test")
        dataset_test = datasets.load_dataset(self.dataset_name)["test"]
        self.assertEqual(len(dataset_fds_test), len(dataset_test))

    def test_multiple_partitioners(self):
        """Test if the dataset works when multiple partitioners are specified."""
        num_train_partitions = 100
        num_test_partitions = 100
        dataset_fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={"train": num_train_partitions, "test": num_test_partitions},
        )
        dataset_test_partition0 = dataset_fds.load_partition(0, "test")

        dataset = datasets.load_dataset(self.dataset_name)
        self.assertEqual(
            len(dataset_test_partition0), len(dataset["test"]) // num_test_partitions
        )

if __name__ == "__main__":
    unittest.main()
