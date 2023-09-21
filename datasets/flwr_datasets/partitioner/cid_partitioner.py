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
"""Cid partitioner class that works with Hugging Face Datasets."""
from typing import Dict, List

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class CidPartitioner(Partitioner):
    """Partitioner for dataset that can be divided by reference to clients."""

    def __init__(
        self,
        partition_by: str,
    ):
        super().__init__()
        self._index_to_cid: Dict[int, str] = {}
        self._partition_by = partition_by
        self._idx_to_rows: Dict[int, List[int]] = {}

    def _create_int_idx_to_cid(self):
        """Create a mapping from int indices to unique client ids.

        Client ids come from the columns specified in `partition_by`.
        """
        unique_cid = self._dataset.unique(self._partition_by)
        self._index_to_cid = dict(zip(range(len(unique_cid)), unique_cid))

    def _save_partition_indexing(self, idx: int, rows: List[int]):
        """Store the rows corresponding to the partition of idx.

        It should be used only after the `load_partition` is used.
        """
        self._idx_to_rows[idx] = rows

    def load_partition(self, idx: int) -> datasets.Dataset:
        """Load a single partition corresponding to a single CID.

        The choice of the partition is based on unique integers assigned to each cid.

        Parameters
        ----------
        idx: int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition: Dataset
            single dataset partition
        """
        if len(self._index_to_cid) == 0:
            self._create_int_idx_to_cid()

        return self._dataset.filter(
            lambda row: row[self._partition_by] == self._index_to_cid[idx]
        )

    @property
    def index_to_cid(self) -> Dict[int, str]:
        """Index to corresponding cid from the dataset property."""
        return self._index_to_cid

    @index_to_cid.setter
    def index_to_cid(self, value: Dict[int, str]) -> None:
        raise AttributeError("Setting the index_to_cid dictionary is not allowed.")
