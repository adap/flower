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
from typing import Dict, Union, cast

from flwr_datasets.partitioner import IidPartitioner, Partitioner

tested_datasets = [
    "mnist",
    "cifar10",
    "fashion_mnist",
    "sasha/dog-food",
    "zh-plus/tiny-imagenet",
]


def _instantiate_partitioners(
    partitioners: Union[int, Partitioner, Dict[str, int], Dict[str, Partitioner]]
) -> Dict[str, Partitioner]:
    """Transform the partitioners from the initial format to instantiated objects.

    Parameters
    ----------
    partitioners: Dict[str, int]
        Partitioners specified as split to the number of partitions format.

    Returns
    -------
    partitioners: Dict[str, Partitioner]
        Partitioners specified as split to Partitioner object.
    """
    instantiated_partitioners: Dict[str, Partitioner] = {}
    if isinstance(partitioners, int):
        # We assume that the int regards IidPartitioner specified for the train set
        instantiated_partitioners["train"] = IidPartitioner(num_partitions=partitioners)
    elif isinstance(partitioners, Partitioner):
        # We assume that the Partitioner was specified for the train set
        instantiated_partitioners["train"] = partitioners
    elif isinstance(partitioners, Dict):
        # dict_first_value = list(partitioners.values())[0]
        # Dict[str, Partitioner]
        if all(isinstance(val, Partitioner) for val in partitioners.values()):
            # No need to do anything
            instantiated_partitioners = cast(Dict[str, Partitioner], partitioners)
        # Dict[str, int]
        elif all(isinstance(val, int) for val in partitioners.values()):
            for split_name, num_partitions in partitioners.items():
                assert isinstance(num_partitions, int)
                instantiated_partitioners[split_name] = IidPartitioner(
                    num_partitions=num_partitions
                )
        else:
            raise ValueError("Incorrect type of the 'partitioners' encountered.")
    return instantiated_partitioners


def _check_if_dataset_tested(dataset: str) -> None:
    """Check if the dataset is in the narrowed down list of the tested datasets."""
    if dataset not in tested_datasets:
        warnings.warn(
            f"The currently tested dataset are {tested_datasets}. Given: {dataset}.",
            stacklevel=1,
        )
