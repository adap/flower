"""Resplitter class for Flower Datasets."""
import collections
from typing import Dict, List, Tuple

import datasets
from datasets import Dataset, DatasetDict


class MergeResplitter:
    """Create a new dataset splits according to the `resplit_strategy`.

    The dataset comes with some predefined splits e.g. "train", "valid" and "test". This
    class allows you to create a new dataset with splits created according to your needs
    specified in `resplit_strategy`.

    Parameters
    ----------
    resplit_strategy: ResplitStrategy
        Dictionary with keys - tuples of the current split names to values - the desired
        split names
    """

    def __init__(
        self,
        resplit_strategy: Dict[Tuple[str, ...], str],
    ) -> None:
        self._resplit_strategy: Dict[Tuple[str, ...], str] = resplit_strategy
        self._check_duplicate_desired_splits()

    def __call__(self, dataset: DatasetDict) -> DatasetDict:
        """Resplit the dataset according to the `resplit_strategy`."""
        self._check_correct_keys_in_resplit_strategy(dataset)
        return self.resplit(dataset)

    def resplit(self, dataset: DatasetDict) -> DatasetDict:
        """Resplit the dataset according to the `resplit_strategy`."""
        resplit_dataset = {}
        for divided_from__list, divide_to in self._resplit_strategy.items():
            datasets_from_list: List[Dataset] = []
            for divide_from in divided_from__list:
                datasets_from_list.append(dataset[divide_from])
            if len(datasets_from_list) > 1:
                resplit_dataset[divide_to] = datasets.concatenate_datasets(
                    datasets_from_list
                )
            else:
                resplit_dataset[divide_to] = datasets_from_list[0]
        return datasets.DatasetDict(resplit_dataset)

    def _check_correct_keys_in_resplit_strategy(self, dataset: DatasetDict) -> None:
        """Check if the keys in resplit_strategy are existing dataset splits."""
        dataset_keys = dataset.keys()
        specified_dataset_keys = self._resplit_strategy.keys()
        for key_list in specified_dataset_keys:
            for key in key_list:
                if key not in dataset_keys:
                    raise ValueError(
                        f"The given dataset key '{key}' is not present in the given "
                        f"dataset object. Make sure to use only the keywords that are "
                        f"available in your dataset."
                    )

    def _check_duplicate_desired_splits(self) -> None:
        """Check for duplicate desired split names."""
        desired_splits = list(self._resplit_strategy.values())
        duplicates = [
            item
            for item, count in collections.Counter(desired_splits).items()
            if count > 1
        ]
        if duplicates:
            print(f"Duplicate desired split name '{duplicates[0]}' in resplit strategy")
            raise ValueError(
                f"Duplicate desired split name '{duplicates[0]}' in resplit strategy"
            )
