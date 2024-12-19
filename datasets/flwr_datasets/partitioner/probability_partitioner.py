# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Probability partitioner class that works with Hugging Face Datasets."""


from typing import Dict, List, Optional, Union

import numpy as np

import datasets
from flwr_datasets.common.typing import NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner


class ProbabilityPartitioner(Partitioner):
    def __init__(
        self,
        probabilities: NDArrayFloat,
        partition_by: str,
        # If we want to do the preassigned then we can still keep the min_partition_size
        min_partition_size: int = 10,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._probabilities = probabilities
        self._num_partitions = probabilities.shape[0]
        self._check_num_partitions_greater_than_zero()
        self._partition_by = partition_by
        self._min_partition_size = min_partition_size
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._unique_classes: Optional[Union[List[int], List[str]]] = None
        self._partition_id_to_indices: Dict[int, List[int]] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single partition of a dataset
        """
        # The partitioning is done lazily - only when the first partition is
        # requested. Only the first call creates the indices assignments for all the
        # partition indices.
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return

        labels = np.asarray(self.dataset[self._partition_by])
        unique_label_to_indices = {}
        unique_label_to_size = {}
        self._unique_classes = self.dataset.unique(self._partition_by)
        self._num_unique_classes = len(self._unique_classes)

        for unique_label in self._unique_classes:
            unique_label_to_indices[unique_label] = np.where(labels == unique_label)[0]
            unique_label_to_size[unique_label] = len(
                unique_label_to_indices[unique_label]
            )

        self._partition_id_to_indices = {
            partition_id: [] for partition_id in range(self._num_partitions)
        }

        for unique_label in self._unique_classes:
            probabilities_per_label = self._probabilities[:, unique_label]
            split_sizes = (
                unique_label_to_size[unique_label] * probabilities_per_label
            ).astype(int)
            cumsum_division_numbers = np.cumsum(split_sizes)
            indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]
            split_indices = np.split(
                unique_label_to_indices[unique_label], indices_on_which_split
            )
            for partition_id in range(self._num_partitions):
                self._partition_id_to_indices[partition_id].extend(
                    split_indices[partition_id]
                )

        self._partition_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """Test num_partition left sides correctness."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")
