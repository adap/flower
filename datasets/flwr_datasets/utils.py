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
from typing import Dict, Optional, Tuple, Union, cast

from flwr_datasets.common import Resplitter
from flwr_datasets.partitioner import IidPartitioner, Partitioner
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
    partitioners: Dict[str, Union[Partitioner, int]]
        Dataset split to the Partitioner or a number of IID partitions.

    Returns
    -------
    partitioners: Dict[str, Partitioner]
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
