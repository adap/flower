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
"""Class constrained partitioner class that works with Hugging Face Datasets."""
import warnings
from typing import Dict, List, Optional, Union

import numpy as np

import datasets
from flwr_datasets.partitioner import Partitioner


class ClassConstrainedPartitioner(Partitioner):
    """
    """

    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        num_classes_per_partition: int,
        first_class_deterministic_assignment: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._num_classes_per_partition = num_classes_per_partition
        self._first_class_deterministic_assignment = (
            first_class_deterministic_assignment
        )
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)

        # Utility attributes
        self._partition_id_to_indices: Dict[int, List[int]] = {}
        self._partition_id_to_unique_labels: Dict[int, List[int]] = {
            pid: [] for pid in range(self._num_partitions)
        }
        self._unique_labels: Union[List[int], List[str]] = []
        # Count in how many partitions the label is used
        self._unique_label_to_times_used_counter: Dict[int, int] = {}
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
        self._determine_partition_id_to_unique_labels()
        self._determine_unique_label_to_times_used()

        # unique_label_to_indices  = {}
        labels = np.asarray(self.dataset[self._partition_by])
        for partition_id in range(self._num_partitions):
            self._partition_id_to_indices[partition_id] = []

        for unique_label in self._unique_labels:
            if self._unique_label_to_times_used_counter[unique_label] == 0:
                warnings.warn(
                    f"Label: {unique_label} will NOT be used due to the chosen configuration. If you want to ensure that each label is used set 'first_class_deterministic_assignment=True'",
                    stacklevel=1,
                )
                continue
            # Get the indices in the original dataset where the y == unique_label
            unique_label_to_indices = np.where(labels == unique_label)[0]

            split_unique_labels_to_indices = np.array_split(
                unique_label_to_indices,
                self._unique_label_to_times_used_counter[unique_label],
            )

            split_index = 0
            for partition_id in range(self._num_partitions):
                if unique_label in self._partition_id_to_unique_labels[partition_id]:
                    self._partition_id_to_indices[partition_id].extend(
                        split_unique_labels_to_indices[split_index]
                    )
                    split_index += 1

        if self._shuffle:
            for indices in self._partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)

        self._partition_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _determine_partition_id_to_unique_labels(self):
        self._unique_labels = self.dataset.unique(self._partition_by)
        num_unique_classes = len(self._unique_labels)

        if self._first_class_deterministic_assignment:
            for partition_id in range(self._num_partitions):
                label = partition_id % num_unique_classes
                self._partition_id_to_unique_labels[partition_id].append(label)

                while (
                    len(self._partition_id_to_unique_labels[partition_id])
                    < self._num_classes_per_partition
                ):
                    label = self._rng.choice(self._unique_labels, size=1)[0]
                    if label not in self._partition_id_to_unique_labels[partition_id]:
                        self._partition_id_to_unique_labels[partition_id].append(label)

        else:
            for partition_id in range(self._num_partitions):
                labels = self._rng.choice(
                    self._unique_labels,
                    size=self._num_classes_per_partition,
                    replace=False,
                ).tolist()
                self._partition_id_to_unique_labels[partition_id] = labels

    def _determine_unique_label_to_times_used(self):
        for unique_label in self._unique_labels:
            self._unique_label_to_times_used_counter[unique_label] = 0
        for unique_labels in self._partition_id_to_unique_labels.values():
            for unique_label in unique_labels:
                self._unique_label_to_times_used_counter[unique_label] += 1
