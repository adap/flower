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

import unittest

import pytest
from parameterized import parameterized, parameterized_class

import datasets
from datasets import DatasetDict, concatenate_datasets
from flwr_datasets.federated_dataset import FederatedDataset


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

    def test_resplit_dataset_into_one(self) -> None:
        """Test resplit into a single dataset."""
        dataset = datasets.load_dataset(self.dataset_name)
        dataset_length = sum([len(ds) for ds in dataset.values()])
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={"train": 100},
            resplitter={("train", self.test_split): "full"},
        )
        full = fds.load_full("full")
        self.assertEqual(dataset_length, len(full))

    # pylint: disable=protected-access
    def test_resplit_dataset_to_change_names(self) -> None:
        """Test resplitter to change the names of the partitions."""
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={"new_train": 100},
            resplitter={
                ("train",): "new_train",
                (self.test_split,): "new_" + self.test_split,
            },
        )
        _ = fds.load_partition(0, "new_train")
        assert fds._dataset is not None
        self.assertEqual(
            set(fds._dataset.keys()), {"new_train", "new_" + self.test_split}
        )

    def test_resplit_dataset_by_callable(self) -> None:
        """Test resplitter to change the names of the partitions."""

        def resplit(dataset: DatasetDict) -> DatasetDict:
            return DatasetDict(
                {
                    "full": concatenate_datasets(
                        [dataset["train"], dataset[self.test_split]]
                    )
                }
            )

        fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": 100}, resplitter=resplit
        )
        full = fds.load_full("full")
        dataset = datasets.load_dataset(self.dataset_name)
        dataset_length = sum([len(ds) for ds in dataset.values()])
        self.assertEqual(len(full), dataset_length)


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
        with pytest.warns(UserWarning):
            FederatedDataset(dataset="food101", partitioners={"train": 100})

    def test_cannot_use_the_old_split_names(self) -> None:
        """Test if the initial split names can not be used."""
        dataset = datasets.load_dataset("mnist")
        sum([len(ds) for ds in dataset.values()])
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": 100},
            resplitter={("train", "test"): "full"},
        )
        with self.assertRaises(ValueError):
            fds.load_partition(0, "train")


if __name__ == "__main__":
    unittest.main()
