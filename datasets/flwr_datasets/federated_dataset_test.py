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
"""Federated Dataset tests."""
# pylint: disable=W0212, C0103


import unittest
from typing import Dict

import pytest
from parameterized import parameterized, parameterized_class

import datasets
from flwr_datasets.federated_dataset import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, Partitioner


@parameterized_class(
    [
        {"dataset_name": "mnist", "test_split": "test"},
        {"dataset_name": "cifar10", "test_split": "test"},
        {"dataset_name": "fashion_mnist", "test_split": "test"},
        {"dataset_name": "sasha/dog-food", "test_split": "test"},
        {"dataset_name": "zh-plus/tiny-imagenet", "test_split": "valid"},
    ]
)
class RealDatasetsFederatedDatasetsTrainTest(unittest.TestCase):
    """Test Real Dataset (MNIST, CIFAR10) in FederatedDatasets."""

    dataset_name = ""
    test_split = ""

    @parameterized.expand(  # type: ignore
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
    def test_load_partition_size(self, _: str, train_num_partitions: int) -> None:
        """Test if the partition size is correct based on the number of partitions."""
        dataset_fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": train_num_partitions}
        )
        dataset_partition0 = dataset_fds.load_partition(0, "train")
        dataset = datasets.load_dataset(self.dataset_name)
        self.assertEqual(
            len(dataset_partition0), len(dataset["train"]) // train_num_partitions
        )

    def test_load_full(self) -> None:
        """Test if the load_full works with the correct split name."""
        dataset_fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": 100}
        )
        dataset_fds_test = dataset_fds.load_full(self.test_split)
        dataset_test = datasets.load_dataset(self.dataset_name)[self.test_split]
        self.assertEqual(len(dataset_fds_test), len(dataset_test))

    def test_multiple_partitioners(self) -> None:
        """Test if the dataset works when multiple partitioners are specified."""
        num_train_partitions = 100
        num_test_partitions = 100
        dataset_fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={
                "train": num_train_partitions,
                self.test_split: num_test_partitions,
            },
        )
        dataset_test_partition0 = dataset_fds.load_partition(0, self.test_split)

        dataset = datasets.load_dataset(self.dataset_name)
        self.assertEqual(
            len(dataset_test_partition0),
            len(dataset[self.test_split]) // num_test_partitions,
        )


class PartitionersSpecificationForFederatedDatasets(unittest.TestCase):
    """Test the specifications of partitioners for FederatedDatasets."""

    dataset_name = "cifar10"
    test_split = "test"

    def test_partitioner_as_int_produces_dict_of_partitioners(self) -> None:
        """Test specifying partitioner as an integer creates a dictionary."""
        num_train_partitions = 100
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=num_train_partitions,
        )
        self.assertIsInstance(fds._partitioners, dict)

    def test_partitioner_as_int_produces_single_partitioner_for_train_data(
        self,
    ) -> None:
        """Test if int partitioner creates a single partitioner for the train set."""
        num_train_partitions = 100
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=num_train_partitions,
        )
        self.assertTrue(len(fds._partitioners) == 1 and "train" in fds._partitioners)

    def test_partitioner_as_int_produces_IidPartitioner(self) -> None:
        """Test if int partitioner creates IidPartitioner for the training set."""
        num_train_partitions = 100
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=num_train_partitions,
        )
        self.assertIsInstance(fds._partitioners["train"], IidPartitioner)

    def test_single_partitioner_gets_assigned_to_train_set(self) -> None:
        """Test if IidPartitioner gets assigned to the train data set."""
        num_train_partitions = 100
        iid_partitioner = IidPartitioner(num_partitions=num_train_partitions)
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=iid_partitioner,
        )
        self.assertTrue(len(fds._partitioners) == 1 and "train" in fds._partitioners)

    def test_single_partitioner_given_is_the_same_object(self) -> None:
        """Test if single partitioner is passed directly."""
        num_train_partitions = 100
        iid_partitioner = IidPartitioner(num_partitions=num_train_partitions)
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=iid_partitioner,
        )
        self.assertIs(fds._partitioners["train"], iid_partitioner)

    def test_dict_of_partitioners_given_is_the_same_in_FederatedDataset(self) -> None:
        """Test if a dictionary of partitioners is passed directly."""
        num_train_partitions = 100
        num_test_partitions = 100
        partitioners: Dict[str, Partitioner] = {
            "train": IidPartitioner(num_partitions=num_train_partitions),
            "test": IidPartitioner(num_partitions=num_test_partitions),
        }
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=partitioners,
        )
        self.assertIs(fds._partitioners, partitioners)

    def test_dict_str_int_produces_correct_partitioners(self) -> None:
        """Test if dict partitioners have the same keys."""
        num_train_partitions = 100
        num_test_partitions = 100
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={
                "train": num_train_partitions,
                "test": num_test_partitions,
            },
        )
        self.assertTrue(
            len(fds._partitioners) == 2
            and "train" in fds._partitioners
            and "test" in fds._partitioners
        )


class IncorrectUsageFederatedDatasets(unittest.TestCase):
    """Test incorrect usages in FederatedDatasets."""

    def test_no_partitioner_for_split(self) -> None:  # pylint: disable=R0201
        """Test using load_partition with missing partitioner."""
        dataset_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})

        with pytest.raises(ValueError):
            dataset_fds.load_partition(0, "test")

    def test_no_split_in_the_dataset(self) -> None:  # pylint: disable=R0201
        """Test using load_partition with non-existent split name."""
        dataset_fds = FederatedDataset(
            dataset="mnist", partitioners={"non-existent-split": 100}
        )

        with pytest.raises(ValueError):
            dataset_fds.load_partition(0, "non-existent-split")

    def test_unsupported_dataset(self) -> None:  # pylint: disable=R0201
        """Test creating FederatedDataset for unsupported dataset."""
        with pytest.raises(ValueError):
            FederatedDataset(dataset="food101", partitioners={"train": 100})


if __name__ == "__main__":
    unittest.main()
