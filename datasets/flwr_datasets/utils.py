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
from typing import Dict

from flwr_datasets.partitioner import IidPartitioner, Partitioner

tested_vision_image_classification_datasets = [
    "mnist",
    "cifar10",
    "fashion_mnist",
    "sasha/dog-food",
    "zh-plus/tiny-imagenet",
]
tested_tabular_classification_datasets = ["hitorilabs/iris"]
tested_uncategorized_datasets = ["scikit-learn/iris"]
tested_datasets = [
    *tested_vision_image_classification_datasets,
    *tested_tabular_classification_datasets,
    *tested_uncategorized_datasets,
]


def _instantiate_partitioners(partitioners: Dict[str, int]) -> Dict[str, Partitioner]:
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
    for split_name, num_partitions in partitioners.items():
        instantiated_partitioners[split_name] = IidPartitioner(
            num_partitions=num_partitions
        )
    return instantiated_partitioners


def _check_if_dataset_tested(dataset: str) -> None:
    """Check if the dataset is in the narrowed down list of the tested datasets."""
    if dataset not in tested_datasets:
        warnings.warn(
            f"The currently tested dataset are {tested_datasets}. Given: {dataset}. "
            f"Note that unsupported datasets might cause errors.",
            stacklevel=1,
        )
