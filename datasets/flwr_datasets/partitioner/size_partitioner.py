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
"""DataQuantityPartitioner class."""
from typing import Callable, Dict, List, Union

import numpy as np

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class SizePartitioner(Partitioner):
    """Base class for the deterministic size partitioning schemes based on the cid.

    The client with cid has the following relationship regarding the number of samples.
    cid_to_size_fnc(cid) ~ number of samples for cid

    If the function doesn't transform the cid it's a linear correlations of the number
    of sample for cid and the value of cid. For instance, if the IDs range from 1 to M,
    client with ID 1 gets 1 unit of data, client 2 gets 2 units, and so on, up to client
     M which gets M units.

    Note that size corresponding to the cid is deterministic, yet in case of different
    dataset shuffling the assignment of samples to cid will vary.

    Parameters
    ----------
    num_partitions: int
        The total number of partitions that the data will be divided into.
    cid_to_size_fnc: Callable
        Function that defines the relationship between cid and the number of samples.
    """

    def __init__(
        self,
        num_partitions: int,
        cid_to_size_fnc: Callable,  # type: ignore[type-arg]
    ) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions
        self._cid_to_size_fnc = cid_to_size_fnc

        self._cid_to_size: Dict[int, int] = {}
        self._cid_to_indices: Dict[int, List[int]] = {}
        # A flag to perform only a single compute to determine the indices
        self._cid_to_indices_determined = False

    def load_partition(self, idx: int) -> datasets.Dataset:
        """Load a single partition based on the partition index.

        For this partitioner the number of samples is dependent on the partition idx.

        Parameters
        ----------
        idx: int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition: Dataset
            single dataset partition
        """
        # The partitioning is done lazily - only when the first partition is requested.
        # A single run creates the indices assignments for all the partition indices.
        self._determine_cid_to_indices_if_needed()
        return self.dataset.select(self._cid_to_indices[idx])

    @property
    def cid_to_size(self) -> Dict[int, int]:
        """Cid to the number of samples."""
        return self._cid_to_size

    @property
    def cid_to_indices(self) -> Dict[int, List[int]]:
        """Cid to the list of indices."""
        return self._cid_to_indices

    def _determine_cid_to_size(self) -> None:
        """Determine data quantity associated with partition indices."""
        data_division_in_units = self._cid_to_size_fnc(
            np.linspace(start=1, stop=self._num_partitions, num=self._num_partitions)
        )
        total_units: Union[int, float] = data_division_in_units.sum()
        # Normalize the units to get the fraction total dataset
        partition_sizes_as_fraction = data_division_in_units / total_units
        # Calculate the number of samples
        partition_sizes_as_num_of_samples = np.array(
            partition_sizes_as_fraction * len(self.dataset), dtype=np.int64
        )
        # Check if any sample is not allocated because of multiplication with fractions.
        assigned_samples = np.sum(partition_sizes_as_num_of_samples)
        left_unassigned_samples = len(self.dataset) - assigned_samples
        # If there is any sample(s) left unassigned, assign it to the largest partition.
        partition_sizes_as_num_of_samples[-1] += left_unassigned_samples
        for idx, partition_size in enumerate(partition_sizes_as_num_of_samples):
            self._cid_to_size[idx] = partition_size

        self._check_if_cid_to_size_possible()

    def _determine_cid_to_indices_if_needed(self) -> None:
        """Create an assignment of indices to the partition indices.."""
        if self._cid_to_indices_determined is True:
            return
        self._determine_cid_to_size()
        total_samples_assigned = 0
        for idx, quantity in self._cid_to_size.items():
            self._cid_to_indices[idx] = list(
                range(total_samples_assigned, total_samples_assigned + quantity)
            )
            total_samples_assigned += quantity
        self._cid_to_indices_determined = True

    def _check_if_cid_to_size_possible(self) -> None:
        all_positive = all(value >= 1 for value in self.cid_to_size.values())
        if not all_positive:
            raise ValueError(
                f"The given specification of the parameter num_partitions"
                f"={self._num_partitions} for the given dataset results "
                f"in the partitions sizes that are not greater than 0."
            )
