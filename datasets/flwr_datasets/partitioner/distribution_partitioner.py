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

""" Distribution partitioner."""
from typing import Dict, List, Optional

import numpy as np

import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class DistributionPartitioner(Partitioner):  # pylint: disable=R0902
    """Partitioner based on a distribution.

    Inspired from implementations of Li et al.
    "Federated Optimization in Heterogeneous Networks" (2020)
    https://arxiv.org/abs/1812.06127.

    Parameters
    ----------
    distribution_array : numpy.ndarray
        A 2-dimensional numpy array of the probability distribution of samples
        for all labels in all partitions. The array shape should be
        (`num_unique_labels`,
        `num_unique_labels_per_partition*num_partitions/num_unique_labels`),
        such that the first row of the array corresponds to the sample distribution
        of the first unique label (in ascending order). The values may be scaled per
        label such that the sum of the label distributions across all partitions are
        equal (or close to) to the original unpartitioned label distribution
        - see the `rescale` argument.
    num_partitions : int
        The total number of partitions that the data will be divided into.
    num_unique_labels_per_partition : int
        Number of unique labels assigned to a single partition.
    partition_by : str
        Column name of the labels (targets) based on which sampling works.
    preassigned_num_samples_per_label : int
        The minimum number of samples that each label in each partition will have.
    rescale : bool, default=True
        Whether to partition samples according to the values in
        `distribution_array` or rescale based on the original unpartitioned
        class label distribution. `float` values are rounded to the nearest `int`.
    shuffle : bool, default=True
        Whether to randomize the order of samples. Shuffling applied after the
        samples assignment to nodes.
    seed : int, default=42
        Seed used for dataset shuffling. It has no effect if `shuffle` is False.
    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import DistributionPartitioner
    >>> from pprint import pprint
    >>>
    >>> num_clients = 1_000
    >>> num_unique_labels_per_client = 2
    >>> num_unique_labels = 10
    >>>
    >>> # Generate a vector from a log-normal probability distribution
    >>> rng = np.random.default_rng(2024)
    >>> mu, sigma = 0., 2.
    >>> lognormal_distribution = rng.lognormal(
    >>>     mu,
    >>>     sigma,
    >>>     (num_clients*num_unique_labels_per_client),
    >>> )
    >>> lognormal_distribution = lognormal_distribution.reshape((num_unique_labels, -1))
    >>>
    >>> partitioner = DistributionPartitioner(
    >>>     distribution_array=lognormal_distribution,
    >>>     num_partitions=num_clients,
    >>>     num_unique_labels_per_partition=num_unique_labels_per_client,
    >>>     partition_by="label",  # MNIST dataset has a target column `label`
    >>>     preassigned_num_samples_per_label=5,
    >>> )
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    >>> print(partition[0])  # Print the first example
    {'image': <PIL.PngImagePlugin.PngImageFile image mode=L size=28x28 at 0x169DD54D0>,
    'label': 0}
    >>> distributions = {
    >>>                     partition_id: fds
    >>>                                   .load_partition(partition_id=partition_id)
    >>>                                   .to_pandas()['label']
    >>>                                   .value_counts()
    >>>                                   .to_dict()
    >>>                     for partition_id in range(10)
    >>>                 }
    >>> pprint(distributions)
    {0: {0: 40, 1: 5},
     1: {2: 36, 1: 5},
     2: {3: 52, 2: 7},
     3: {3: 14, 4: 6},
     4: {4: 47, 5: 28},
     5: {6: 30, 5: 5},
     6: {6: 19, 7: 11},
     7: {8: 22, 7: 11},
     8: {9: 11, 8: 5},
     9: {0: 124, 9: 13}}
    """

    def __init__(  # pylint: disable=R0913
        self,
        distribution_array: np.ndarray,
        num_partitions: int,
        num_unique_labels_per_partition: int,
        partition_by: str,
        preassigned_num_samples_per_label: int,
        rescale: bool = True,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._distribution_array = distribution_array
        self._num_partitions = num_partitions
        self._num_unique_labels_per_partition = num_unique_labels_per_partition
        self._partition_by = partition_by
        self._preassigned_num_samples_per_label = preassigned_num_samples_per_label
        self._rescale = rescale
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._num_unique_labels: int = None
        self._partition_id_to_indices_determined = False
        self._partition_id_to_indices: Dict[int, List[int]] = {}

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
        self._check_distribution_array_shape_if_needed()
        self._determine_partition_id_to_indices_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions

    def _determine_partition_id_to_indices_if_needed(
        self,
    ) -> None:
        """Create an assignment of indices to the partition indices."""
        if self._partition_id_to_indices_determined:
            return

        # Compute the label distribution from the dataset
        unique_labels = sorted(self.dataset.unique(self._partition_by))
        labels = np.asarray(self.dataset[self._partition_by])
        unique_label_to_indices = {}
        unique_label_distribution = {}

        for unique_label in unique_labels:
            unique_label_to_indices[unique_label] = np.where(labels == unique_label)[0]
            unique_label_distribution[unique_label] = len(
                unique_label_to_indices[unique_label]
            )

        if self._rescale:
            # Compute the normalized distribution for each class label
            self._distribution_array = self._distribution_array / np.sum(
                self._distribution_array, axis=-1, keepdims=True
            )

            # Compute the total preassigned number of samples per label for all labels
            # and partitions. This sum will be subtracted from the label distribution
            # of the original dataset, and added back later. It ensures that
            # (1) each partition will have at least
            #     `self._preassigned_num_samples_per_label`, and
            # (2) there is sufficient indices to sample from the dataset.
            total_preassigned_samples = int(
                self._preassigned_num_samples_per_label
                * self._num_unique_labels_per_partition
                * self._num_partitions
                / self._num_unique_labels
            )

            label_distribution = np.fromiter(
                unique_label_distribution.values(),
                dtype=float,
            )

            # Subtract the preassigned total amount from the label distribution,
            # we'll add these back later.
            label_distribution -= total_preassigned_samples

            # Rescale normalized distribution with the actual label distribution.
            # Each row represents the number of samples to be taken for that class label
            # and the sum of each row equals the total of each class label.
            label_sampling_matrix = np.floor(
                (self._distribution_array * label_distribution[:, np.newaxis])
            ).astype(int)

            # Add back the preassigned total amount
            label_sampling_matrix += self._preassigned_num_samples_per_label
        else:
            label_sampling_matrix = self._distribution_array.astype(int)

        # Create the label sampling dictionary
        label_samples = dict(
            zip(unique_label_distribution.keys(), label_sampling_matrix)
        )

        # Create indices split from dataset
        split_indices_per_label = {}
        for unique_label in unique_labels:
            # Compute cumulative sum of samples to identify splitting points
            cumsum_division_numbers = np.cumsum(label_samples[unique_label])
            split_indices = np.split(
                unique_label_to_indices[unique_label], cumsum_division_numbers
            )
            split_indices_per_label[unique_label] = split_indices

        # Initialize sampling tracker. Keys are the unique class labels.
        # Values are the smallest indices of each array in `label_samples`
        # which will be sampled next. Once a sample is taken from a label/key,
        # increment the value (index) by 1.
        index_tracker = {k: 0 for k in unique_labels}

        # Prepare data structure to store indices assigned to partition ids
        self._partition_id_to_indices = {
            partition_id: [] for partition_id in range(self._num_partitions)
        }

        for partition_id in range(self._num_partitions):
            # Get the `num_unique_labels_per_partition` labels for each partition. Use
            # `numpy.roll` to get indices of adjacent sorted labels for pathological
            # label distributions.
            labels_per_client = np.roll(unique_labels, -partition_id)[
                : self._num_unique_labels_per_partition
            ]
            for label in labels_per_client:
                index_to_sample = index_tracker[label]
                self._partition_id_to_indices[partition_id].extend(
                    split_indices_per_label[label][index_to_sample]
                )
                index_tracker[label] += 1

        # Shuffle the indices to avoid datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in self._partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)
        self._partition_id_to_indices_determined = True

    def _check_distribution_array_shape_if_needed(self) -> None:
        """Test distribution array shape correctness."""
        if not self._partition_id_to_indices_determined:
            if not isinstance(self._distribution_array, np.ndarray):
                raise TypeError("Input must be a numpy array.")

            if self._distribution_array.ndim != 2:
                raise ValueError("The distribution array is not 2-dimensional.")

            self._num_unique_labels = len(self.dataset.unique(self._partition_by))
            num_columns = (
                self._num_unique_labels_per_partition
                * self._num_partitions
                / self._num_unique_labels
            )
            if self._distribution_array.shape != (self._num_unique_labels, num_columns):
                raise ValueError(
                    f"The distribution array shape is {self._distribution_array.shape},"
                    f" but it should be ({self._num_unique_labels}, {num_columns})."
                )

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
