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
from typing import List, Optional, Dict, Union

import numpy as np

import datasets
from flwr_datasets.common.typing import NDArrayInt, NDArrayFloat
from flwr_datasets.partitioner import Partitioner


class InnerProbabilityPartitioner(Partitioner):
    def __init__(
            self,
            partition_by: str,
            probabilities: NDArrayFloat,
            partition_sizes: Optional[Union[NDArrayInt, List[int]]] = None,
            shuffle: bool = True,
            seed: Optional[int] = 42,
    ) -> None:
        # todo: when the size are 0 then divide evenly
        # todo: preassignment of n_min samples for the chosen classes
        super().__init__()
        self._partition_by = partition_by
        self._probabilities = probabilities
        self._partition_sizes = _instantiate_partition_sizes(partition_sizes)
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        self._unique_classes: Optional[Union[List[int], List[str]]] = None
        self._num_unique_classes: Optional[int] = None
        self._num_partitions = len(self._partition_sizes)
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
        self._check_partition_sizes_correctness_if_needed()
        self._check_the_sum_of_partition_sizes()
        self._determine_num_unique_classes_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_num_partitions_correctness_if_needed()
        self._check_partition_sizes_correctness_if_needed()
        self._check_the_sum_of_partition_sizes()
        self._determine_num_unique_classes_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self._num_partitions

    def _determine_partition_id_to_indices_if_needed(
            self,
    ) -> None:  # pylint: disable=R0914
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return
        labels = np.asarray(self.dataset[self._partition_by])
        assert self._num_unique_classes is not None
        unique_label_to_indices = {}
        unique_label_to_size = {}
        for unique_label in self._unique_classes:
            unique_label_to_indices[unique_label] = np.where(labels == unique_label)[0]
            unique_label_to_size[unique_label] = len(unique_label_to_indices[unique_label])


        self._partition_id_to_indices = {}
        partition_id_to_left_to_allocate = {}
        for partition_id in range(self._num_partitions):
            self._partition_id_to_indices[partition_id] = []
            partition_id_to_left_to_allocate[partition_id] = self._partition_sizes[partition_id]

        not_full_partition_ids = [partition_id for partition_id in range(self._num_partitions) if self._partition_sizes[partition_id] != 0]
        class_priors = self._probabilities.copy()

        while np.sum(list(partition_id_to_left_to_allocate.values())) != 0:
            # Choose a partition
            current_partition_id = self._rng.choice(not_full_partition_ids)
            # If current partition is full resample a client
            if partition_id_to_left_to_allocate[current_partition_id] == 0:
                # When the partition is full, exclude it from the sampling list
                not_full_partition_ids.pop(
                    not_full_partition_ids.index(current_partition_id)
                )
                continue
            partition_id_to_left_to_allocate[current_partition_id] -= 1
            # Access the label distribution of the chosen client
            current_probabilities = class_priors[current_partition_id]
            # print("current_probabilities")
            # print(current_probabilities)
            while True:
                curr_class = self._rng.choice(
                    list(range(self._num_unique_classes)), p=current_probabilities
                )
                # Redraw class label if there are no samples left to be allocated from
                # that class
                if unique_label_to_size[curr_class] == 0:
                    #print("curr_class")
                    #print(curr_class)
                    #print("curr_partition_id")
                    #print(current_partition_id)
                    #print("class_priors before")
                    #print(class_priors)
                    # Class got exhausted, set probabilities to 0
                    class_priors[:, curr_class] = 0
                    # Renormalize such that the probability sums to 1
                    row_sums = class_priors.sum(axis=1, keepdims=True)
                    #print("indices of rows equal zero")
                    partition_ids_with_zero_prob = np.where(row_sums == 0)[0]
                    for partition_id_with_zero_prob in partition_ids_with_zero_prob:
                        if partition_id_to_left_to_allocate[partition_id_with_zero_prob] > 0:
                            raise ValueError(f"The partition id {partition_id_with_zero_prob} requires allocation of {partition_id_to_left_to_allocate[partition_id_with_zero_prob]} more sample however the samples from the specified distributions have alread been assigned. Adjust the probabilities of the given partition sizes to make the sampling possible.")
                    #print()
                    #print("class_priors after")
                    #print(class_priors)
                    #print("partition_id_to_left_to_allocate")
                    #print(partition_id_to_left_to_allocate)
                    #print("row_sums")
                    #print(row_sums)
                    class_priors = class_priors / row_sums
                    # Adjust the current_probabilities (it won't sum up to 1 otherwise)
                    current_probabilities = class_priors[current_partition_id]
                    continue
                unique_label_to_size[curr_class] -= 1
                # Store sample index at the empty array cell
                index = partition_id_to_left_to_allocate[current_partition_id]
                self._partition_id_to_indices[current_partition_id].append(unique_label_to_indices[curr_class][
                    unique_label_to_size[curr_class]
                ])
                break


        # Shuffle the indices if the shuffle is True.
        # Note that the samples from this partitioning do not necessarily require
        # shuffling, the order should exhibit consecutive samples.
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
                    "The number of partitions needs to be smaller or equal to "
                    " the number of samples in the dataset."
                )

    def _check_partition_sizes_correctness_if_needed(self) -> None:
        """Test partition_sizes when the dataset is given (in load_partition)."""
        if not self._partition_id_to_indices_determined:
            if sum(self._partition_sizes) > self.dataset.num_rows:
                raise ValueError(
                    "The sum of the `partition_sizes` needs to be smaller or equal to "
                    "the number of samples in the dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """Test num_partition left sides correctness."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")

    def _determine_num_unique_classes_if_needed(self) -> None:
        self._unique_classes = self.dataset.unique(self._partition_by)
        assert self._unique_classes is not None
        self._num_unique_classes = len(self._unique_classes)

    def _check_the_sum_of_partition_sizes(self) -> None:
        if np.sum(self._partition_sizes) != len(self.dataset):
            warnings.warn(
                "The sum of the partition_sizes does not sum to the whole "
                "dataset size. Make sure that is the desired behavior.",
                stacklevel=1,
            )


def _instantiate_partition_sizes(
        partition_sizes: Union[List[int], NDArrayInt]
) -> NDArrayInt:
    """Transform list to the ndarray of ints if needed."""
    if isinstance(partition_sizes, List):
        partition_sizes = np.asarray(partition_sizes)
    elif isinstance(partition_sizes, np.ndarray):
        pass
    else:
        raise ValueError(
            f"The type of partition_sizes is incorrect. Given: "
            f"{type(partition_sizes)}"
        )

    if not all(partition_sizes >= 0):
        raise ValueError("The samples numbers must be greater or equal to zero.")
    return partition_sizes
