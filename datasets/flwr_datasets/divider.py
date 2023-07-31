"""Divider class for flwr datasets"""
from typing import Optional, Union, List, Dict

import datasets
from datasets import Dataset, DatasetDict


class Divider:
    allowed_federated_keys = ["federated", "centralized", "final-test"]

    def __init__(
            self,
            dataset: DatasetDict,
            division_strategy: Union[Dict[str, Optional[str]], str],
    ) -> None:
        """
        Divide and merge the dataset according to the strategy in `division_strategy`.

        Create the meaningful division for the FL "federated", "centralized",
        "final-test" parts. The "federated" part will be used to create the data for
        each  client (on edge/local data). The "centralized" will be used for
        centralized evaluation. And the "final-test" is the counterpart of the test in
        centralized ML to estimate the generalization ability of the model.

        {"federated": "train", "centralized": "test"} - means that give the dataset
        that is {"train": train_dataset, "test": test_dataset} the federated clients
        will be created from the train_dataset and the centralized dataset will be
        created from test_dataset.

        If you want to merge the "train" and "test" parts of the dataset
        that is {"train": train_dataset, "test": test_dataset} so that you can decide on
        the % of the partitions used "federated+centralized": "train+test".

        Specifying e.g. the "final-test": None has the same effect as not mentioning it.
        So the {"federated": "train"} has the same effect as {"federated": "train",
        "centralized": None, "final-test": None}

        IN FUTURE: Support also Dict[str: Optional[list[str]] such that it can be e.g.
        {"federated": ["train", "test"]}

        Parameters
        ----------
        dataset: DatasetDict
            Dataset that can be directly loaded from HuggingFace.
        division_strategy: Union[Dict[str: Optional[str]], str]
            Federated key name to `dataset` key name. Federated keys are: "federated",
                "centralized" and "final-test". The `dataset` key name are the keys of
                the dataset which are typically "train", "test", "valid". The
                "federated" key has to be used, other are optional. Alternatively string
                "merge-all" to create {"federated+centralized+final-test":
                all-dataset-keys}.
        """

        self._dataset: Union[Dataset, DatasetDict] = dataset
        self._division_strategy: Dict[str: Optional[str]] = division_strategy
        self._check_federated_in_division_strategy()
        self._check_correct_federated_keys_in_division_strategy()
        self._check_correct_dataset_keys_in_division_strategy()
        # self.divided_dataset: DatasetDict = self.divide()

    def divide(self) -> DatasetDict:
        """Divide the dataset according to the give `division_strategy`."""
        if self._division_strategy == "merge-all":
            keys = list(self._dataset.keys())
            keys_str = ""
            for key in keys:
                keys_str += key + "+"
            keys_str = keys_str[:-1]
            self._division_strategy = {"federated+centralized+final-test": keys_str}
        divided_dataset = {}
        for division, strategy in self._division_strategy.items():
            if strategy is None:
                pass
            strategy_kws = strategy.split("+")
            dataset_list: List[Dataset] = []
            for kw in strategy_kws:
                dataset_list.append(self._dataset[kw])
            if len(dataset_list) > 1:
                divided_dataset[division] = datasets.concatenate_datasets(dataset_list)
            else:
                divided_dataset[division] = dataset_list[0]
        return datasets.DatasetDict(divided_dataset)

    def _check_federated_in_division_strategy(self) -> None:
        """Check if federated key is given.

        It doesn't make sense to have the `division_strategy` without it.
        """
        if isinstance(self._division_strategy,
                      str) and self._division_strategy == "merge-all":
            return
        else:
            key_groups = self._division_strategy.keys()
            for key_group in key_groups:
                keys = key_group.split("+")
                for key in keys:
                    if "federated" in key:
                        return
            raise ValueError("The `division_strategy` misses the 'federated' key.")

    def _check_correct_federated_keys_in_division_strategy(self) -> None:
        if isinstance(self._division_strategy,
                      str) and self._division_strategy == "merge-all":
            return
        else:
            specified_federated_keys = []
            for key in self._division_strategy.keys():
                specified_federated_keys.extend(key.split("+"))
            if len(specified_federated_keys) > 3:
                raise ValueError(
                    "The federated keys in the `division_strategy` cannot repeat. "
                    "Please make sure to specify them correctly.")
            for key in specified_federated_keys:
                if key not in self.allowed_federated_keys:
                    raise ValueError(
                        f"The given federated key {key} is not allowed to specify the "
                        "federated divisions.")

    def _check_correct_dataset_keys_in_division_strategy(self) -> None:
        if isinstance(self._division_strategy,
                      str) and self._division_strategy == "merge-all":
            return
        specified_dataset_keys = []
        for dataset_key in self._division_strategy.values():
            specified_dataset_keys.extend(dataset_key.split("+"))

        if len(specified_dataset_keys) > 3:
            raise ValueError(
                "The dataset keys in the `division_strategy` cannot repeat. "
                "Please make sure to specify them correctly.")

        for key in specified_dataset_keys:
            if key not in self._dataset.keys():
                raise ValueError(
                    f"The given dataset key {key} is not present in the `dataset`. "
                    f"Make sure to use only the keywords that are available in your "
                    f"dataset.")
