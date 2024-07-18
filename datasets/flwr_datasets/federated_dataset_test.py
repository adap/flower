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

import numpy as np
import pytest
from parameterized import parameterized, parameterized_class

import datasets
from datasets import Dataset, DatasetDict, concatenate_datasets
from flwr_datasets.federated_dataset import FederatedDataset
from flwr_datasets.mock_utils_test import _load_mocked_dataset
from flwr_datasets.partitioner import IidPartitioner, NaturalIdPartitioner, Partitioner

mocked_datasets = ["cifar100", "svhn", "sentiment140", "speech_commands"]


@parameterized_class(
    ("dataset_name", "test_split", "subset"),
    [
        # Downloaded
        # #Image datasets
        ("mnist", "test", ""),
        ("cifar10", "test", ""),
        ("fashion_mnist", "test", ""),
        ("sasha/dog-food", "test", ""),
        ("zh-plus/tiny-imagenet", "valid", ""),
        # Text
        ("scikit-learn/adult-census-income", None, ""),
        # Mocked
        # #Image
        ("cifar100", "test", ""),
        # Note: there's also the extra split and full_numbers subset
        ("svhn", "test", "cropped_digits"),
        # Text
        ("sentiment140", "test", ""),  # aka twitter
        # Audio
        ("speech_commands", "test", "v0.01"),
    ],
)
class BaseFederatedDatasetsTest(unittest.TestCase):
    """Test Real/Mocked Datasets used in FederatedDatasets.

    The setUp method mocks the dataset download via datasets.load_dataset if it is in
    the `mocked_datasets` list.
    """

    dataset_name = ""
    test_split = ""
    subset = ""

    def setUp(self) -> None:
        """Mock the dataset download prior to each method if needed.

        If the `dataset_name` is in the `mocked_datasets` list, then the dataset
        download is mocked.
        """
        if self.dataset_name in mocked_datasets:
            self.patcher = patch("datasets.load_dataset")
            self.mock_load_dataset = self.patcher.start()
            self.mock_load_dataset.return_value = _load_mocked_dataset(
                self.dataset_name, [200, 100], ["train", self.test_split], self.subset
            )

    def tearDown(self) -> None:
        """Clean up after the dataset mocking."""
        if self.dataset_name in mocked_datasets:
            patch.stopall()

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
        # Compute the actual partition sizes
        partition_sizes = []
        for node_id in range(train_num_partitions):
            partition_sizes.append(len(dataset_fds.load_partition(node_id, "train")))

        #  Create the expected sizes of partitions
        dataset = datasets.load_dataset(self.dataset_name)
        full_train_length = len(dataset["train"])
        expected_sizes = []
        default_partition_size = full_train_length // train_num_partitions
        mod = full_train_length % train_num_partitions
        for i in range(train_num_partitions):
            expected_sizes.append(default_partition_size + (1 if i < mod else 0))
        self.assertEqual(partition_sizes, expected_sizes)

    def test_load_split(self) -> None:
        """Test if the load_split works with the correct split name."""
        if self.test_split is None:
            return
        dataset_fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": 100}
        )
        dataset_fds_test = dataset_fds.load_split(self.test_split)
        dataset_test = datasets.load_dataset(self.dataset_name)[self.test_split]
        self.assertEqual(len(dataset_fds_test), len(dataset_test))

    def test_multiple_partitioners(self) -> None:
        """Test if the dataset works when multiple partitioners are specified."""
        if self.test_split is None:
            return
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
        fds = FederatedDataset(dataset=self.dataset_name, partitioners={"train": 10})
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
        if self.test_split is None:
            return
        dataset = datasets.load_dataset(self.dataset_name)
        dataset_length = sum([len(ds) for ds in dataset.values()])
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={"train": 100},
            preprocessor={"full": ("train", self.test_split)},
        )
        full = fds.load_split("full")
        self.assertEqual(dataset_length, len(full))

    # pylint: disable=protected-access
    def test_resplit_dataset_to_change_names(self) -> None:
        """Test preprocessor to change the names of the partitions."""
        if self.test_split is None:
            return
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={"new_train": 100},
            preprocessor={
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
        """Test preprocessor to change the names of the partitions."""
        if self.test_split is None:
            return

        def resplit(dataset: DatasetDict) -> DatasetDict:
            return DatasetDict(
                {
                    "full": concatenate_datasets(
                        [dataset["train"], dataset[self.test_split]]
                    )
                }
            )

        fds = FederatedDataset(
            dataset=self.dataset_name, partitioners={"train": 100}, preprocessor=resplit
        )
        full = fds.load_split("full")
        dataset = datasets.load_dataset(self.dataset_name)
        dataset_length = sum([len(ds) for ds in dataset.values()])
        self.assertEqual(len(full), dataset_length)


class ShufflingResplittingOnArtificialDatasetTest(unittest.TestCase):
    """Test shuffling and resplitting using small artificial dataset.

    The purpose of this class is to ensure the order of samples remains as expected.

    The load_dataset method is mocked and the artificial dataset is returned.
    """

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
        train = fds.load_split("train")
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
        train = fds.load_split("train")
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
            preprocessor=resplit,
            shuffle=True,
        )
        train = fds.load_split("train")
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


natural_id_datasets = [
    "flwrlabs/femnist",
]


@parameterized_class(
    ("dataset_name", "test_split", "subset", "partition_by"),
    [
        ("flwrlabs/femnist", "", "", "writer_id"),
    ],
)
class NaturalIdPartitionerIntegrationTest(unittest.TestCase):
    """General FederatedDataset tests with NaturalIdPartitioner."""

    dataset_name = ""
    test_split = ""
    subset = ""
    partition_by = ""

    def test_if_the_partitions_have_unique_values(self) -> None:
        """Test if each partition has a single unique id value."""
        fds = FederatedDataset(
            dataset=self.dataset_name,
            partitioners={
                "train": NaturalIdPartitioner(partition_by=self.partition_by)
            },
        )
        for partition_id in range(fds.partitioners["train"].num_partitions):
            partition = fds.load_partition(partition_id)
            unique_ids_in_partition = list(set(partition[self.partition_by]))
            self.assertEqual(len(unique_ids_in_partition), 1)


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
            preprocessor={"full": ("train", "test")},
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
        # Ensure all keys are the same in both rows
        if set(row1.keys()) != set(row2.keys()):
            return False

        # Compare values for each key
        for key in row1:
            if key == "audio":
                # Special handling for 'audio' key
                if not all(
                    [
                        np.array_equal(row1[key]["array"], row2[key]["array"]),
                        row1[key]["path"] == row2[key]["path"],
                        row1[key]["sampling_rate"] == row2[key]["sampling_rate"],
                    ]
                ):
                    return False
            elif row1[key] != row2[key]:
                # Direct comparison for other keys
                return False

    return True


if __name__ == "__main__":
    unittest.main()
