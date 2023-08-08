import unittest

import datasets

from federated_dataset import FederatedDataset


class MNISTFederatedDatasets(unittest.TestCase):
    """Test MNIST dataset used in FederatedDatasets."""

    def test_load_partition_size(self):
        dataset_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
        dataset_partition0 = dataset_fds.load_partition(0, "train")
        dataset = datasets.load_dataset("mnist")
        self.assertEqual(len(dataset_partition0), len(dataset["train"]) // 100)

    def test_load_full(self):
        dataset_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
        dataset_fds_test = dataset_fds.load_full("test")
        dataset_test = datasets.load_dataset("mnist")["test"]
        self.assertEqual(len(dataset_fds_test), len(dataset_test))

    def test_multiple_partitioners(self):
        num_train_partitions = 100
        num_test_partitions = 100
        dataset_fds = FederatedDataset(dataset="mnist",
                                       partitioners={"train": num_train_partitions,
                                                     "test":
                                                         num_test_partitions})
        dataset_test_partition0 = dataset_fds.load_partition(0, "test")

        dataset = datasets.load_dataset("mnist")
        self.assertEqual(len(dataset_test_partition0),
                         len(dataset["test"]) // num_test_partitions)


class CIFAR10FederatedDatasets(unittest.TestCase):
    """Test CIFAR10 dataset used in FederatedDatasets."""

    def test_load_partition_size(self):
        dataset_fds = FederatedDataset(dataset="cifar10", partitioners={"train": 100})
        dataset_partition0 = dataset_fds.load_partition(0, "train")
        dataset = datasets.load_dataset("cifar10")
        self.assertEqual(len(dataset_partition0), len(dataset["train"]) // 100)

    def test_load_full(self):
        dataset_fds = FederatedDataset(dataset="cifar10", partitioners={"train": 100})
        dataset_fds_test = dataset_fds.load_full("test")
        dataset_test = datasets.load_dataset("cifar10")["test"]
        self.assertEqual(len(dataset_fds_test), len(dataset_test))

    def test_multiple_partitioners(self):
        num_train_partitions = 100
        num_test_partitions = 10
        dataset_fds = FederatedDataset(dataset="cifar10",
                                       partitioners={"train": num_train_partitions,
                                                     "test":
                                                         num_test_partitions})
        dataset_test_partition0 = dataset_fds.load_partition(0, "test")

        dataset = datasets.load_dataset("cifar10")
        self.assertEqual(len(dataset_test_partition0),
                         len(dataset["test"]) // num_test_partitions)


# class IncorrectUsageFederatedDatasets(unittest.TestCase):
#     """Test incorrect usages in FederatedDatasets."""
#
#     def test_unsupported_dataset_exception(self):


if __name__ == '__main__':
    unittest.main()
