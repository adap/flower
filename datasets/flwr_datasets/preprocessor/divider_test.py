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
"""Divider tests."""

import unittest
from typing import Dict, Union

from parameterized import parameterized_class

from datasets import Dataset, DatasetDict
from flwr_datasets.preprocessor import Divider


@parameterized_class(
    ("divide_config", "divide_split", "drop_remaining_splits", "split_name_to_size"),
    [
        # Standard config that sums to one
        (
            {"train_1": 0.25, "train_2": 0.75},
            "train",
            False,
            {"train_1": 10, "train_2": 30, "valid": 20, "test": 40},
        ),
        # As the first use case but drop the remaining splits
        (
            {"train_1": 0.25, "train_2": 0.75},
            "train",
            True,
            {"train_1": 10, "train_2": 30},
        ),
        # Split without ratios
        (
            {"train_1": 15, "train_2": 25},
            "train",
            False,
            {"train_1": 15, "train_2": 25, "valid": 20, "test": 40},
        ),
        # Split does not sum to 1.0
        (
            {"a": 0.2, "b": 0.4},
            "valid",
            False,
            {"a": 4, "b": 8, "train": 40, "test": 40},
        ),
        # Completely custom names
        (
            {"test_a": 0.2, "asdfasdfsa": 0.4},
            "test",
            False,
            {"test_a": 8, "asdfasdfsa": 16, "valid": 20, "train": 40},
        ),
        # Mirror copies of the first example but using multiple split
        (
            {"train": {"train_1": 0.25, "train_2": 0.75}},
            None,
            False,
            {"train_1": 10, "train_2": 30, "valid": 20, "test": 40},
        ),
        # Resplitting multiple splits
        (
            {
                "train": {"train_1": 0.25, "train_2": 0.75},
                "valid": {"valid_1": 0.4, "valid_2": 0.6},
            },
            None,
            False,
            {"train_1": 10, "train_2": 30, "valid_1": 8, "valid_2": 12, "test": 40},
        ),
    ],
)
class TestDivider(unittest.TestCase):
    """Divider tests."""

    divide_config: Union[
        Dict[str, float],
        Dict[str, int],
        Dict[str, Dict[str, float]],
        Dict[str, Dict[str, int]],
    ]
    divide_split: str
    drop_remaining_splits: bool
    split_name_to_size: Dict[str, int]

    def setUp(self) -> None:
        """Set up the dataset with 3 splits for tests."""
        self.dataset_dict = DatasetDict(
            {
                "train": Dataset.from_dict({"data": list(range(40))}),
                "valid": Dataset.from_dict({"data": list(range(40, 60))}),
                "test": Dataset.from_dict({"data": list(range(60, 100))}),
            }
        )

    def test_resplitting_correct_new_split_names(self) -> None:
        """Test if resplitting produces requested new splits."""
        divider = Divider(
            self.divide_config, self.divide_split, self.drop_remaining_splits
        )
        resplit_dataset = divider(self.dataset_dict)
        new_keys = set(resplit_dataset.keys())
        self.assertEqual(set(self.split_name_to_size.keys()), new_keys)

    def test_resplitting_correct_new_split_sizes(self) -> None:
        """Test if resplitting produces correct sizes of splits."""
        divider = Divider(
            self.divide_config, self.divide_split, self.drop_remaining_splits
        )
        resplit_dataset = divider(self.dataset_dict)
        split_to_size = {
            split_name: len(split) for split_name, split in resplit_dataset.items()
        }
        self.assertEqual(self.split_name_to_size, split_to_size)


class TestDividerIncorrectUseCases(unittest.TestCase):
    """Divider tests."""

    def setUp(self) -> None:
        """Set up the dataset with 3 splits for tests."""
        self.dataset_dict = DatasetDict(
            {
                "train": Dataset.from_dict({"data": list(range(40))}),
                "valid": Dataset.from_dict({"data": list(range(40, 60))}),
                "test": Dataset.from_dict({"data": list(range(60, 100))}),
            }
        )

    def test_doubling_names_in_config(self) -> None:
        """Test if resplitting raises when the same name in config is detected."""
        divide_config = {"train": {"new_train": 0.5}, "valid": {"new_train": 0.3}}
        divide_split = None
        drop_remaining_splits = False

        with self.assertRaises(ValueError):
            divider = Divider(divide_config, divide_split, drop_remaining_splits)
            _ = divider(self.dataset_dict)

    def test_duplicate_names_in_config_and_dataset_split_names_multisplit(self) -> None:
        """Test if resplitting raises when the name collides with the old name."""
        divide_config = {"train": {"valid": 0.5}}
        divide_split = None
        drop_remaining_splits = False
        divider = Divider(divide_config, divide_split, drop_remaining_splits)
        with self.assertRaises(ValueError):
            _ = divider(self.dataset_dict)

    def test_duplicate_names_in_config_and_dataset_split_names_single_split(
        self,
    ) -> None:
        """Test if resplitting raises when the name collides with the old name."""
        divide_config = {"valid": 0.5}
        divide_split = "train"
        drop_remaining_splits = False
        divider = Divider(divide_config, divide_split, drop_remaining_splits)
        with self.assertRaises(ValueError):
            _ = divider(self.dataset_dict)

    def test_fraction_sum_up_to_more_than_one_multisplit(self) -> None:
        """Test if resplitting raises when fractions sum up to > 1.0 ."""
        divide_config = {"train": {"train_1": 0.5, "train_2": 0.7}}
        divide_split = None
        drop_remaining_splits = False
        divider = Divider(divide_config, divide_split, drop_remaining_splits)
        with self.assertRaises(ValueError):
            _ = divider(self.dataset_dict)

    def test_fraction_sum_up_to_more_than_one_single_split(self) -> None:
        """Test if resplitting raises when fractions sum up to > 1.0 ."""
        divide_config = {"train_1": 0.5, "train_2": 0.7}
        divide_split = "train"
        drop_remaining_splits = False
        divider = Divider(divide_config, divide_split, drop_remaining_splits)
        with self.assertRaises(ValueError):
            _ = divider(self.dataset_dict)

    def test_sample_sizes_sum_up_to_more_than_dataset_size_single_split(self) -> None:
        """Test if resplitting raises when samples size sum up to > len(datset) ."""
        divide_config = {"train": {"train_1": 20, "train_2": 25}}
        divide_split = None
        drop_remaining_splits = False
        divider = Divider(divide_config, divide_split, drop_remaining_splits)
        with self.assertRaises(ValueError):
            _ = divider(self.dataset_dict)

    def test_sample_sizes_sum_up_to_more_than_dataset_size_multisplit(self) -> None:
        """Test if resplitting raises when samples size sum up to > len(datset) ."""
        divide_config = {"train_1": 20, "train_2": 25}
        divide_split = "train"
        drop_remaining_splits = False
        divider = Divider(divide_config, divide_split, drop_remaining_splits)
        with self.assertRaises(ValueError):
            _ = divider(self.dataset_dict)

    def test_too_small_size_values_create_empty_dataset_single_split(self) -> None:
        """Test if resplitting raises when fraction creates empty dataset."""
        divide_config = {"train": {"train_1": 0.2, "train_2": 0.0001}}
        divide_split = None
        drop_remaining_splits = False
        divider = Divider(divide_config, divide_split, drop_remaining_splits)
        with self.assertRaises(ValueError):
            _ = divider(self.dataset_dict)

    def test_too_small_size_values_create_empty_dataset_multisplit(self) -> None:
        """Test if resplitting raises when fraction creates empty dataset."""
        divide_config = {"train_1": 0.2, "train_2": 0.0001}
        divide_split = "train"
        drop_remaining_splits = False
        divider = Divider(divide_config, divide_split, drop_remaining_splits)
        with self.assertRaises(ValueError):
            _ = divider(self.dataset_dict)


if __name__ == "__main__":
    unittest.main()
