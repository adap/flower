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
"""Preprocessor tests."""


import unittest
from typing import Dict, Tuple

import pytest

from datasets import Dataset, DatasetDict
from flwr_datasets.resplitter.merge_resplitter import MergeResplitter


class TestResplitter(unittest.TestCase):
    """Preprocessor tests."""

    def setUp(self) -> None:
        """Set up the dataset with 3 splits for tests."""
        self.dataset_dict = DatasetDict(
            {
                "train": Dataset.from_dict({"data": [1, 2, 3]}),
                "valid": Dataset.from_dict({"data": [4, 5]}),
                "test": Dataset.from_dict({"data": [6]}),
            }
        )

    def test_resplitting_train_size(self) -> None:
        """Test if resplitting for just renaming keeps the lengths correct."""
        strategy: Dict[str, Tuple[str, ...]] = {"new_train": ("train",)}
        resplitter = MergeResplitter(strategy)
        new_dataset = resplitter(self.dataset_dict)
        self.assertEqual(len(new_dataset["new_train"]), 3)

    def test_resplitting_valid_size(self) -> None:
        """Test if resplitting for just renaming keeps the lengths correct."""
        strategy: Dict[str, Tuple[str, ...]] = {"new_valid": ("valid",)}
        resplitter = MergeResplitter(strategy)
        new_dataset = resplitter(self.dataset_dict)
        self.assertEqual(len(new_dataset["new_valid"]), 2)

    def test_resplitting_test_size(self) -> None:
        """Test if resplitting for just renaming keeps the lengths correct."""
        strategy: Dict[str, Tuple[str, ...]] = {"new_test": ("test",)}
        resplitter = MergeResplitter(strategy)
        new_dataset = resplitter(self.dataset_dict)
        self.assertEqual(len(new_dataset["new_test"]), 1)

    def test_resplitting_train_the_same(self) -> None:
        """Test if resplitting for just renaming keeps the dataset the same."""
        strategy: Dict[str, Tuple[str, ...]] = {"new_train": ("train",)}
        resplitter = MergeResplitter(strategy)
        new_dataset = resplitter(self.dataset_dict)
        self.assertTrue(
            datasets_are_equal(self.dataset_dict["train"], new_dataset["new_train"])
        )

    def test_combined_train_valid_size(self) -> None:
        """Test if the resplitting that combines the datasets has correct size."""
        strategy: Dict[str, Tuple[str, ...]] = {
            "train_valid_combined": ("train", "valid")
        }
        resplitter = MergeResplitter(strategy)
        new_dataset = resplitter(self.dataset_dict)
        self.assertEqual(len(new_dataset["train_valid_combined"]), 5)

    def test_resplitting_test_with_combined_strategy_size(self) -> None:
        """Test if the resplitting that combines the datasets has correct size."""
        strategy: Dict[str, Tuple[str, ...]] = {
            "train_valid_combined": ("train", "valid"),
            "test": ("test",),
        }
        resplitter = MergeResplitter(strategy)
        new_dataset = resplitter(self.dataset_dict)
        self.assertEqual(len(new_dataset["test"]), 1)

    def test_invalid_resplit_strategy_exception_message(self) -> None:
        """Test if the resplitting raises error when non-existing split is given."""
        strategy: Dict[str, Tuple[str, ...]] = {
            "new_train": ("invalid_split",),
            "new_test": ("test",),
        }
        resplitter = MergeResplitter(strategy)
        with self.assertRaisesRegex(
            ValueError, "The given dataset key 'invalid_split' is not present"
        ):
            resplitter(self.dataset_dict)

    def test_nonexistent_split_in_strategy(self) -> None:
        """Test if the exception is raised when the nonexistent split name is given."""
        strategy: Dict[str, Tuple[str, ...]] = {"new_split": ("nonexistent_split",)}
        resplitter = MergeResplitter(strategy)
        with self.assertRaisesRegex(
            ValueError, "The given dataset key 'nonexistent_split' is not present"
        ):
            resplitter(self.dataset_dict)

    def test_duplicate_merge_split_name(self) -> None:  # pylint: disable=R0201
        """Test that the new split names are not the same."""
        strategy: Dict[str, Tuple[str, ...]] = {
            "new_train": ("train", "valid"),
            "test": ("train",),
        }
        with pytest.warns(UserWarning):
            _ = MergeResplitter(strategy)

    def test_empty_dataset_dict(self) -> None:
        """Test that the error is raised when the empty DatasetDict is given."""
        empty_dataset = DatasetDict({})
        strategy: Dict[str, Tuple[str, ...]] = {"new_train": ("train",)}
        resplitter = MergeResplitter(strategy)
        with self.assertRaisesRegex(
            ValueError, "The given dataset key 'train' is not present"
        ):
            resplitter(empty_dataset)


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
