import unittest
from datasets import Dataset, DatasetDict

from divider import Divider


class TestDivider(unittest.TestCase):

    def test_divide_with_valid_strategy(self):
        """Test whether the valid divide strategy works."""
        dataset = DatasetDict({"train": Dataset.from_dict({"data": [1, 2, 3]}),
                               "test": Dataset.from_dict({"data": [4, 5, 6]})})
        division_strategy = {"federated": "train", "centralized": "test"}

        divider = Divider(dataset, division_strategy)
        result = divider.divide()

        self.assertEqual(result["federated"]["data"], [1, 2, 3])
        self.assertEqual(result["centralized"]["data"], [4, 5, 6])

    def test_divide_with_merge_all_strategy(self):
        """Test merge-all strategy."""
        dataset = DatasetDict({"train": Dataset.from_dict({"data": [1, 2, 3]}),
                               "test": Dataset.from_dict({"data": [4, 5, 6]})})

        divider = Divider(dataset, "merge-all")
        result = divider.divide()

        self.assertEqual(result["federated+centralized+final-test"]["data"],
                         [1, 2, 3, 4, 5, 6])

    def test_divide_with_invalid_strategy_missing_federated_key(self):
        """Test missing federated key missing in divide_strategy."""
        dataset = DatasetDict({"train": Dataset.from_dict({"data": [1, 2, 3]}),
                               "test": Dataset.from_dict({"data": [4, 5, 6]})})
        division_strategy = {"centralized": "test"}  # Missing 'federated' key

        with self.assertRaises(ValueError):
            divider = Divider(dataset, division_strategy)

    def test_divide_with_invalid_strategy_wrong_key(self):
        """Test incorrect dataset key in divide_strategy."""
        dataset = DatasetDict({"train": Dataset.from_dict({"data": [1, 2, 3]}),
                               "test": Dataset.from_dict({"data": [4, 5, 6]})})
        division_strategy = {
            "federated": "unknown"}  # 'unknown' key not present in the dataset

        with self.assertRaises(ValueError):
            divider = Divider(dataset, division_strategy)


if __name__ == "__main__":
    unittest.main()
