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
"""Natural id partitioner class that works with Hugging Face Datasets."""


from typing import Dict

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class NaturalIdPartitioner(Partitioner):
    """Partitioner for dataset that can be divided by a reference to id in dataset."""

    def __init__(
        self,
        partition_by: str,
    ):
        super().__init__()
        self._partition_id_to_natural_id: Dict[int, str] = {}
        self._partition_by = partition_by

    def _create_int_partition_id_to_natural_id(self) -> None:
        """Create a mapping from int indices to unique client ids from dataset.

        Natural ids come from the column specified in `partition_by`.
        """
        unique_natural_ids = self.dataset.unique(self._partition_by)
        self._partition_id_to_natural_id = dict(
            zip(range(len(unique_natural_ids)), unique_natural_ids)
        )

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single partition corresponding to a single `partition_id`.

        The choice of the partition is based on unique integers assigned to each
        natural id present in the dataset in the `partition_by` column.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        if len(self._partition_id_to_natural_id) == 0:
            self._create_int_partition_id_to_natural_id()

        return self.dataset.filter(
            lambda row: row[self._partition_by]
            == self._partition_id_to_natural_id[partition_id]
        )

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        if len(self._partition_id_to_natural_id) == 0:
            self._create_int_partition_id_to_natural_id()
        return len(self._partition_id_to_natural_id)

    @property
    def partition_id_to_natural_id(self) -> Dict[int, str]:
        """Node id to corresponding natural id present.

        Natural ids are the unique values in `partition_by` column in dataset.
        """
        return self._partition_id_to_natural_id

    # pylint: disable=R0201
    @partition_id_to_natural_id.setter
    def partition_id_to_natural_id(self, value: Dict[int, str]) -> None:
        raise AttributeError(
            "Setting the partition_id_to_natural_id dictionary is not allowed."
        )
