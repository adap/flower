# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Utils tests."""
import unittest
from typing import Dict, List, Tuple, Union

from parameterized import parameterized_class

from datasets import Dataset, DatasetDict
from flwr_datasets.utils import divide_dataset


@parameterized_class(
    (
        "divide",
        "sizes",
    ),
    [
        ((0.2, 0.8), [8, 32]),
        ([0.2, 0.8], [8, 32]),
        ({"train": 0.2, "test": 0.8}, [8, 32]),
        # Not full dataset
        ([0.2, 0.1], [8, 4]),
        ((0.2, 0.1), [8, 4]),
        ({"train": 0.2, "test": 0.1}, [8, 4]),
    ],
)
class UtilsTests(unittest.TestCase):
    """Utils tests."""

    divide: Union[List[float], Tuple[float, ...], Dict[str, float]]
    sizes: Tuple[int]

    def setUp(self) -> None:
        """Set up a dataset."""
        self.dataset = Dataset.from_dict({"data": range(40)})

    def test_correct_sizes(self) -> None:
        """Test correct size of the division."""
        divided_dataset = divide_dataset(self.dataset, self.divide)
        if isinstance(divided_dataset, (list, tuple)):
            lengths = [len(split) for split in divided_dataset]
        else:
            lengths = [len(split) for split in divided_dataset.values()]

        self.assertEqual(lengths, self.sizes)

    def test_correct_return_types(self) -> None:
        """Test correct types of the divided dataset based on the config."""
        divided_dataset = divide_dataset(self.dataset, self.divide)
        if isinstance(self.divide, (list, tuple)):
            self.assertIsInstance(divided_dataset, list)
        else:
            self.assertIsInstance(divided_dataset, DatasetDict)


if __name__ == "__main__":
    unittest.main()
