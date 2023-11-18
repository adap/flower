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
# pylint: disable=W0212, C0103, C0206


import unittest
from typing import Dict, Union
from unittest.mock import Mock, patch

import pytest
from parameterized import parameterized, parameterized_class

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
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

    def test_no_need_for_split_keyword_if_one_partitioner(self) -> None:
        """Test if partitions got with and without split args are the same."""
        fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
        partition_loaded_with_no_split_arg = fds.load_partition(0)
        partition_loaded_with_verbose_split_arg = fds.load_partition(0, "train")
        self.assertTrue(
            datasets_are_equal(
                partition_loaded_with_no_split_arg,
                partition_loaded_with_verbose_split_arg,
            )
        )

    def test_resplit_dataset_into_one(self) -> None:
        """Test resplit into a single dataset."""
        dataset = datasets.load_dataset(self.dataset_name)
        dataset_length = sum([len(ds) for ds in dataset.values()])
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={"train": 100},
            resplitter={"full": ("train", self.test_split)},
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
                "new_train": ("train",),
                "new_" + self.test_split: (self.test_split,),
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


class ArtificialDatasetTest(unittest.TestCase):
    """Test using small artificial dataset, mocked load_dataset."""

    # pylint: disable=no-self-use
    def _dummy_setup(self, train_rows: int = 10, test_rows: int = 5) -> DatasetDict:
        """Create a dummy DatasetDict with train, test splits."""
        data_train = {
            "features": list(range(train_rows)),
            "labels": list(range(100, 100 + train_rows)),
        }
        data_test = {
            "features": [200] + [201] * (test_rows - 1),
            "labels": [202] + [203] * (test_rows - 1),
        }
        train_dataset = Dataset.from_dict(data_train)
        test_dataset = Dataset.from_dict(data_test)
        return DatasetDict({"train": train_dataset, "test": test_dataset})

    @patch("datasets.load_dataset")
    def test_shuffling_applied(self, mock_func: Mock) -> None:
        """Test if argument is used."""
        dummy_ds = self._dummy_setup()
        mock_func.return_value = dummy_ds

        expected_result = dummy_ds.shuffle(seed=42)["train"]["features"]
        fds = FederatedDataset(
            dataset="does-not-matter", partitioners={"train": 10}, shuffle=True, seed=42
        )
        train = fds.load_full("train")
        # This should be shuffled
        result = train["features"]

        self.assertEqual(expected_result, result)

    @patch("datasets.load_dataset")
    def test_shuffling_not_applied(self, mock_func: Mock) -> None:
        """Test if argument is not used."""
        dummy_ds = self._dummy_setup()
        mock_func.return_value = dummy_ds

        expected_result = dummy_ds["train"]["features"]
        fds = FederatedDataset(
            dataset="does-not-matter",
            partitioners={"train": 10},
            shuffle=False,
        )
        train = fds.load_full("train")
        # This should not be shuffled
        result = train["features"]

        self.assertEqual(expected_result, result)

    @patch("datasets.load_dataset")
    def test_shuffling_before_to_resplitting_applied(self, mock_func: Mock) -> None:
        """Check if the order is met and if the shuffling happens."""

        def resplit(dataset: DatasetDict) -> DatasetDict:
            #  "Move" the last sample from test to train
            return DatasetDict(
                {
                    "train": concatenate_datasets(
                        [dataset["train"], dataset["test"].select([0])]
                    ),
                    "test": dataset["test"].select(range(1, dataset["test"].num_rows)),
                }
            )

        dummy_ds = self._dummy_setup()
        mock_func.return_value = dummy_ds

        expected_result = concatenate_datasets(
            [dummy_ds["train"].shuffle(42), dummy_ds["test"].shuffle(42).select([0])]
        )["features"]
        fds = FederatedDataset(
            dataset="does-not-matter",
            partitioners={"train": 10},
            resplitter=resplit,
            shuffle=True,
        )
        train = fds.load_full("train")
        # This should not be shuffled
        result = train["features"]

        self.assertEqual(expected_result, result)


class PartitionersSpecificationForFederatedDatasets(unittest.TestCase):
    """Test the specifications of partitioners for `FederatedDataset`."""

    dataset_name = "cifar10"
    test_split = "test"

    def test_dict_of_partitioners_passes_partitioners(self) -> None:
        """Test if partitioners are passed directly (no recreation)."""
        num_train_partitions = 100
        num_test_partitions = 100
        partitioners: Dict[str, Union[Partitioner, int]] = {
            "train": IidPartitioner(num_partitions=num_train_partitions),
            "test": IidPartitioner(num_partitions=num_test_partitions),
        }
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=partitioners,
        )

        self.assertTrue(
            all(fds._partitioners[key] == partitioners[key] for key in partitioners)
        )

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

    def test_mixed_type_partitioners_passes_instantiated_partitioners(self) -> None:
        """Test if an instantiated partitioner is passed directly."""
        num_train_partitions = 100
        num_test_partitions = 100
        partitioners: Dict[str, Union[Partitioner, int]] = {
            "train": IidPartitioner(num_partitions=num_train_partitions),
            "test": num_test_partitions,
        }
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=partitioners,
        )
        self.assertIs(fds._partitioners["train"], partitioners["train"])

    def test_mixed_type_partitioners_creates_from_int(self) -> None:
        """Test if an IidPartitioner partitioner is created."""
        num_train_partitions = 100
        num_test_partitions = 100
        partitioners: Dict[str, Union[Partitioner, int]] = {
            "train": IidPartitioner(num_partitions=num_train_partitions),
            "test": num_test_partitions,
        }
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners=partitioners,
        )
        self.assertTrue(
            isinstance(fds._partitioners["test"], IidPartitioner)
            and fds._partitioners["test"]._num_partitions == num_test_partitions
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
        with pytest.warns(UserWarning):
            FederatedDataset(dataset="food101", partitioners={"train": 100})

    def test_cannot_use_the_old_split_names(self) -> None:
        """Test if the initial split names can not be used."""
        dataset = datasets.load_dataset("mnist")
        sum([len(ds) for ds in dataset.values()])
        fds = FederatedDataset(
            dataset="mnist",
            partitioners={"train": 100},
            resplitter={"full": ("train", "test")},
        )
        with self.assertRaises(ValueError):
            fds.load_partition(0, "train")


def datasets_are_equal(ds1: Dataset, ds2: Dataset) -> bool:
    """Check if two Datasets have the same values."""
    # Check if both datasets have the same length
    if len(ds1) != len(ds2):
        return False

    # Iterate over each row and check for equality
    for row1, row2 in zip(ds1, ds2):
        if row1 != row2:
            return False

    return True


if __name__ == "__main__":
    unittest.main()
