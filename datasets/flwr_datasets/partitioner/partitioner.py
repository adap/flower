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
"""Partitioner class that works with HuggingFace Dataset."""
from abc import ABC, abstractmethod
from typing import Optional

import datasets
from datasets import Dataset


class Partitioner(ABC):
    """The base partitioner class that enables obtaining federated partitions.

    The initialization is intended to take all necessary arguments such that the call to
    the `load_partition` method can be use the same for all partitioners.
    """

    def __init__(self):
        self._dataset: Optional[Dataset] = None

    @property
    def dataset(self):
        """Dataset property."""
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset):
        if self._dataset is None:
            self._dataset = value
        else:
            raise Warning(
                "The dataset should be assigned only once to the partitioner."
                "This operation might also wipe out the saved references to the "
                "created partitions (in case the partitioning scheme needs to create "
                "the full partitioning also in order to return a single partition)."
            )

    @abstractmethod
    def load_partition(self, partition_index: int) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        partition_index: int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition: Dataset
            single dataset partition
        """
        raise NotImplementedError

    def _check_if_dataset_assigned(self):
        """Check if dataset is assigned - it should be prior to using load_partition."""
        if self._dataset is None:
            raise ValueError(
                "The dataset field should be set before using the load_partition."
            )
