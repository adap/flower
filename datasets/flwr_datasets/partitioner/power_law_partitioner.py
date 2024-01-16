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
"""Power Law partitioner."""
from typing import Dict, List, Optional, Union

import numpy as np

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class PowerLawPartitioner(Partitioner):  # pylint: disable=R0902
    """Partitioner based on Power Law distribution and restricting the unique classes.

    Implementation based on Federated Optimization in Heterogeneous Networks
    https://arxiv.org/abs/1812.06127.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    partition_by : str
        Column name of the labels (targets) based on which sampling works.
    num_labels_per_partition : int
        Number of unique labels assigned to a single partition.
    mean : float
        Mean to log-normal distribution.
    sigma:  float
        Sigma to log-normal distribution.
    min_partition_size : int
        The minimum number of samples that each partitions will have (the sampling
        process is repeated if any partition is too small).
    n_classes_to_preassign : int
        Number of unique classes that are assigned prior to the probabilistic-based
        assignment.
    n_samples_per_class_to_preassign : int
        The number of sample per unique class that will be assigned prior to the
        probabilistic-based assigment.
    shuffle : bool
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to nodes.
    seed : int
        Seed used for dataset shuffling. It has no effect if `shuffle` is False.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import InnerDirichletPartitioner
    >>>
    >>> partitioner = InnerDirichletPartitioner(
    >>>     partition_sizes=[6_000] * 10, partition_by="label", alpha=0.5
    >>> )
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    >>> print(partition[0])  # Print the first example
    {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x127BF6950>,
    'label': 3}
    >>> partition_sizes = [len(fds.load_partition(node_id)) for node_id in range(10)]
    >>> print(sorted(partition_sizes))
    [1762, 2667, 2958, 3034, 4895, 7743, 8054, 9280, 9726, 9855]
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_partitions: int,
        partition_by: str,
        num_labels_per_partition: int,
        mean: float = 0.0,
        sigma: float = 2.0,
        min_partition_size: Optional[int] = None,
        n_classes_to_preassign: int = 2,
        n_samples_per_class_to_preassign: int = 5,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._num_labels_per_partition = num_labels_per_partition
        self._partition_by = partition_by
        self._n_classes_to_preassign = n_classes_to_preassign
        self._n_samples_per_class_to_preassign = n_samples_per_class_to_preassign
        if min_partition_size is None:
            # Note that zero might make problems with the training
            min_partition_size = 0
        self._min_partition_size: int = min_partition_size
        self._mean = mean
        self._sigma = sigma
        self._shuffle = shuffle
        self._seed = seed

        # Utility attributes
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # The attributes below are determined during the first call to load_partition
        self._num_unique_classes: Optional[int] = None
        self._unique_classes: Optional[Union[List[int], List[str]]] = None

        self._node_id_to_indices: Dict[int, List[int]] = {
            node_id: [] for node_id in range(self._num_partitions)
        }
        self._node_id_to_indices_determined = False
        self._label_to_indices: Dict[Union[int, str], List[int]] = {}
        self._label_to_remaining_for_allocation: List[int] = []
        self._class_to_num_allocated_samples: Dict[Union[int, str], int] = {}

        # Specific to Power Law
        # Node id to number of samples left for allocation for that node id
        self._node_id_to_left_to_allocate = None

    def load_partition(self, node_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        node_id : int
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
        self._determine_node_id_to_indices_if_needed()
        return self.dataset.select(self._node_id_to_indices[node_id])

    def _determine_node_id_to_indices_if_needed(self) -> None:  # pylint: disable=R0914
        if self._node_id_to_indices_determined:
            return

        # Prepare helpful structures for further preprocessing
        labels = self.dataset[self._partition_by]
        labels_array = np.array(labels)
        # unique_labels = np.unique(labels_array)
        self._unique_classes = np.unique(labels_array)  # self.dataset.unique(
        # self._partition_by)
        assert self._unique_classes is not None
        self._num_unique_classes = len(self._unique_classes)
        self._label_to_indices = {
            label: np.where(labels_array == label)[0].tolist()
            for label in self._unique_classes
        }
        self._label_to_remaining_for_allocation = [
            len(self._label_to_indices[i]) for i in range(self._num_unique_classes)
        ]
        self._class_to_num_allocated_samples = {
            class_: 0 for class_ in self._unique_classes
        }
        # Preassing samples
        if self._n_classes_to_preassign * self._n_samples_per_class_to_preassign != 0:
            self._preassing_samples()

        # Assign according to "power law"

        probs = self._rng.lognormal(
            self._mean,
            self._sigma,
            (
                self._num_unique_classes,
                int(self._num_partitions / self._num_unique_classes),
                self._num_labels_per_partition,
            ),
        )
        probs = (
            np.array(self._label_to_remaining_for_allocation)[:, np.newaxis, np.newaxis]
            * probs
            / np.sum(probs, (1, 2), keepdims=True)
        )
        for node_id in range(self._num_partitions):
            # Loop through self._num_labels_per_partition to assign samples from these
            # ith_class_on_node works as an offset to determine the classes for the node
            for ith_class_on_node in range(self._num_labels_per_partition):
                class_index = (node_id + ith_class_on_node) % self._num_unique_classes
                num_new_samples = int(
                    probs[
                        class_index,
                        node_id // self._num_unique_classes,
                        ith_class_on_node,
                    ]
                )

                # Add new samples if there are sufficiently enough samples
                if (
                    num_new_samples
                    > self._label_to_remaining_for_allocation[class_index]
                ):
                    continue
                # Determine new indices to add
                start_idx = self._class_to_num_allocated_samples[class_index]
                end_idx = start_idx + num_new_samples
                new_indices = self._label_to_indices[class_index][start_idx:end_idx]
                self._node_id_to_indices[node_id].extend(new_indices)
                self._class_to_num_allocated_samples[class_index] += num_new_samples
                self._label_to_remaining_for_allocation[class_index] -= num_new_samples

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in self._node_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)
        self._node_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        """Test num_partitions when the dataset is given (in load_partition)."""
        if not self._node_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows:
                raise ValueError(
                    "The number of partitions needs to be smaller than the number of "
                    "samples in the dataset."
                )

    def _check_num_partitions_greater_than_zero(self) -> None:
        """Test num_partition left sides correctness."""
        if not self._num_partitions > 0:
            raise ValueError("The number of partitions needs to be greater than zero.")

    def _preassing_samples(self) -> None:
        for node_id in range(self._num_partitions):
            for ith_class_on_node in range(self._n_classes_to_preassign):
                assert self._num_unique_classes is not None
                class_index = (node_id + ith_class_on_node) % self._num_unique_classes
                start_idx = self._class_to_num_allocated_samples[class_index]
                end_idx = (
                    self._class_to_num_allocated_samples[class_index]
                    + self._n_samples_per_class_to_preassign
                )
                self._node_id_to_indices[node_id].extend(
                    self._label_to_indices[class_index][start_idx:end_idx]
                )
                self._class_to_num_allocated_samples[
                    class_index
                ] += self._n_samples_per_class_to_preassign
                self._label_to_remaining_for_allocation[
                    class_index
                ] -= self._n_samples_per_class_to_preassign


if __name__ == "__main__":
    from flwr_datasets import FederatedDataset
    from flwr_datasets.partitioner import PowerLawPartitioner

    partitioner = PowerLawPartitioner(
        num_partitions=10, partition_by="label", num_labels_per_partition=5
    )
    fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    partition = fds.load_partition(0)
    print(partition[0])  # Print the first example
    partition_sizes = [len(fds.load_partition(node_id)) for node_id in range(10)]
    print(sorted(partition_sizes))
