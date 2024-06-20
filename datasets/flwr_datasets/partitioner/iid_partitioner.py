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
"""IID partitioner class that works with Hugging Face Datasets."""


from typing import Any, Dict, List, Optional

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class IidPartitioner(Partitioner):
    """Partitioner creates each partition sampled randomly from the dataset.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    """

    def __init__(self, num_partitions: int) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a single IID partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions

    @property
    def partition_id_to_indices(self) -> Dict[int, List[int]]:
        """Representation of the result of partitioning."""
        self._determine_partition_id_to_indices_if_needed()
        return self._partition_id_to_indices

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return
        for partition_id in range(self.num_partitions):
            # adapted based on the shard implementation from datasets.Dataset.shard
            div = len(self.dataset) // self.num_partitions
            mod = len(self.dataset) % self.num_partitions
            start = div * partition_id + min(partition_id, mod)
            end = start + div + (1 if partition_id < mod else 0)
            indices = list(range(start, end))
            self._partition_id_to_indices[partition_id] = indices
        self._partition_id_to_indices_determined = True

    def to_config(self) -> Dict[str, Any]:
        """Create a configuration (a dictionary) representing the partitioner.

        This method is used in `to_config_file`.

        Returns
        -------
        partitioner_representation: Dict[str, Any]
            Parameters representing a partitioner.
        """
        config = {
            "_target_": f"{self.__class__.__module__}.{self.__class__.__name__}."
            f"from_config",
            "num_partitions": self.num_partitions,
        }
        return config

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        partition_id_to_indices: Optional[Dict[int, List[int]]] = None,
    ) -> "IidPartitioner":
        """Instantiate the partitioner based on the given parameters.

        This method should make sure that the object internal state is correctly
        reflected (when indices mapping is inferred there should be no need for new
        partitioning).

        Parameters
        ----------
        config: Dict[str, Any]
            Representation of the partitioner that
        partition_id_to_indices: Optional[Dict[int, List[int]]]
            Mapping of partition_id to indices that was created when partitioning the
            dataset.

        Returns
        -------
        partitioner: Partitioner
            An instantiated partitioner
        """
        iid = cls(config["num_partitions"])
        if partition_id_to_indices is not None:
            iid._partition_id_to_indices = partition_id_to_indices
            # _partition_id_to_indices_determined needs to be True to avoid
            # re-creation of the indices
            iid._partition_id_to_indices_determined = True
        return iid
