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
"""Utils for FederatedDataset."""


import warnings
from typing import Dict, List, Optional, Tuple, Union, cast

from datasets import Dataset, DatasetDict
from flwr_datasets.partitioner import IidPartitioner, Partitioner
from flwr_datasets.resplitter import Resplitter
from flwr_datasets.resplitter.merge_resplitter import MergeResplitter

tested_datasets = [
    "mnist",
    "cifar10",
    "fashion_mnist",
    "sasha/dog-food",
    "zh-plus/tiny-imagenet",
]


def _instantiate_partitioners(
    partitioners: Dict[str, Union[Partitioner, int]]
) -> Dict[str, Partitioner]:
    """Transform the partitioners from the initial format to instantiated objects.

    Parameters
    ----------
    partitioners : Dict[str, Union[Partitioner, int]]
        Dataset split to the Partitioner or a number of IID partitions.

    Returns
    -------
    partitioners : Dict[str, Partitioner]
        Partitioners specified as split to Partitioner object.
    """
    instantiated_partitioners: Dict[str, Partitioner] = {}
    if isinstance(partitioners, Dict):
        for split, partitioner in partitioners.items():
            if isinstance(partitioner, Partitioner):
                instantiated_partitioners[split] = partitioner
            elif isinstance(partitioner, int):
                instantiated_partitioners[split] = IidPartitioner(
                    num_partitions=partitioner
                )
            else:
                raise ValueError(
                    f"Incorrect type of the 'partitioners' value encountered. "
                    f"Expected Partitioner or int. Given {type(partitioner)}"
                )
    else:
        raise ValueError(
            f"Incorrect type of the 'partitioners' encountered. "
            f"Expected Dict[str, Union[int, Partitioner]]. "
            f"Given {type(partitioners)}."
        )
    return instantiated_partitioners


def _instantiate_resplitter_if_needed(
    resplitter: Optional[Union[Resplitter, Dict[str, Tuple[str, ...]]]]
) -> Optional[Resplitter]:
    """Instantiate `MergeResplitter` if resplitter is merge_config."""
    if resplitter and isinstance(resplitter, Dict):
        resplitter = MergeResplitter(merge_config=resplitter)
    return cast(Optional[Resplitter], resplitter)


def _check_if_dataset_tested(dataset: str) -> None:
    """Check if the dataset is in the narrowed down list of the tested datasets."""
    if dataset not in tested_datasets:
        warnings.warn(
            f"The currently tested dataset are {tested_datasets}. Given: {dataset}.",
            stacklevel=1,
        )


def divide_dataset(
    dataset: Dataset, division: Union[List[float], Tuple[float, ...], Dict[str, float]]
) -> Union[List[Dataset], DatasetDict]:
    """Divide the dataset according to the `division`.

    The division support varying number of splits, which you can name. The splits are
    created from the beginning of the dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset to be divided.
    division: Union[List[float], Tuple[float, ...], Dict[str, float]]
        Configuration specifying how the dataset is divided. Each fraction has to be
        >0 and <=1. They have to sum up to at most 1 (smaller sum is possible).

    Returns
    -------
    divided_dataset : Union[List[Dataset], DatasetDict]
        If `division` is `List` or `Tuple` then `List[Dataset]` is returned else if
        `division` is `Dict` then `DatasetDict` is returned.

    Examples
    --------
    Use `divide_dataset` with division specified as a list.
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.utils import divide_dataset
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> partition = fds.load_partition(0)
    >>> division = [0.8, 0.2]
    >>> train, test = divide_dataset(dataset=partition, division=division)

    Use `divide_dataset` with division specified as a dict.
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.utils import divide_dataset
    >>>
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})
    >>> partition = fds.load_partition(0)
    >>> division = {"train": 0.8, "test": 0.2}
    >>> train_test = divide_dataset(dataset=partition, division=division)
    >>> train, test = train_test["train"], train_test["test"]
    """
    dataset_length = len(dataset)
    ranges = _create_division_indices_ranges(dataset_length, division)
    if isinstance(division, (list, tuple)):
        split_partition: List[Dataset] = []
        for single_range in ranges:
            split_partition.append(dataset.select(single_range))
        return split_partition
    if isinstance(division, dict):
        split_partition_dict: Dict[str, Dataset] = {}
        for split_name, single_range in zip(division.keys(), ranges):
            split_partition_dict[split_name] = dataset.select(single_range)
        return DatasetDict(split_partition_dict)
    raise TypeError(
        f"The type of the `division` should be dict, "
        f"tuple or list but is {type(division)} instead."
    )


def _create_division_indices_ranges(
    dataset_length: int,
    division: Union[List[float], Tuple[float, ...], Dict[str, float]],
) -> List[range]:
    ranges = []
    if isinstance(division, (list, tuple)):
        start_idx = 0
        end_idx = 0
        for fraction in division:
            end_idx += int(dataset_length * fraction)
            ranges.append(range(start_idx, end_idx))
            start_idx += end_idx
    elif isinstance(division, dict):
        ranges = []
        start_idx = 0
        end_idx = 0
        for fraction in division.values():
            end_idx += int(dataset_length * fraction)
            ranges.append(range(start_idx, end_idx))
            start_idx += end_idx
    else:
        TypeError(
            f"The type of the `division` should be dict, "
            f"tuple or list but is {type(division)} instead. "
        )
    return ranges


def _check_division_config_types_correctness(
    division: Union[List[float], Tuple[float, ...], Dict[str, float]]
) -> None:
    if isinstance(division, (list, tuple)):
        if not all(isinstance(x, float) for x in division):
            raise TypeError(
                "List or tuple values of `division` must contain only floats, "
                "other types are not allowed."
            )
    elif isinstance(division, dict):
        if not all(isinstance(x, float) for x in division.values()):
            raise TypeError(
                "Dict values of `division` must be only floats, "
                "other types are not allowed."
            )
    else:
        raise TypeError("`division` must be a list, tuple, or dict.")


def _check_division_config_values_correctness(
    division: Union[List[float], Tuple[float, ...], Dict[str, float]]
) -> None:
    if isinstance(division, (list, tuple)):
        if not all(0 < x <= 1 for x in division):
            raise ValueError(
                "All fractions for the division must be greater than 0 and smaller or "
                "equal to 1."
            )
        fraction_sum_from_list_tuple = sum(division)
        if fraction_sum_from_list_tuple > 1:
            raise ValueError("Sum of fractions for division must not exceed 1.")
        if fraction_sum_from_list_tuple < 1:
            warnings.warn(
                f"Sum of fractions for division is {sum(division)}, which is below 1. "
                f"Make sure that's the desired behavior. Some data will not be used "
                f"in the current specification.",
                stacklevel=1,
            )
    elif isinstance(division, dict):
        values = list(division.values())
        if not all(0 < x <= 1 for x in values):
            raise ValueError(
                "All fractions must be greater than 0 and smaller or equal to 1."
            )
        if sum(values) > 1:
            raise ValueError("Sum of fractions must not exceed 1.")
        if sum(values) < 1:
            warnings.warn(
                f"Sum of fractions in `division` is {values}, which is below 1. "
                f"Make sure that's the desired behavior. Some data will not be used "
                f"in the current specification.",
                stacklevel=1,
            )
    else:
        raise TypeError("`division` must be a list, tuple, or dict.")


def _check_division_config_correctness(
    division: Union[List[float], Tuple[float, ...], Dict[str, float]]
) -> None:
    _check_division_config_types_correctness(division)
    _check_division_config_values_correctness(division)
