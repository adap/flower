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
from typing import Dict, List, Optional, Union

import numpy as np
from flwr_datasets.partitioner.partitioner import Partitioner

import datasets


class DistributionPartitioner(Partitioner):  # pylint: disable=R0902
    """Partitioner based on a distribution.

    Inspired from implementations of Li et al.
    "Federated Optimization in Heterogeneous Networks" (2020)
    https://arxiv.org/abs/1812.06127.

    Parameters
    ----------
    distribution_array : numpy.ndarray
        Sample distribution for all labels in all partitions. The array shape
        must be of dimension `num_unique_labels` x
        `num_unique_labels_per_partition*(num_partitions/num_unique_labels)`.
        The values may be scaled per label such that the sum of the label
        distributions across all partitions equal to the original unpartitioned
        label distribution - see the `rescale` argument. The final per-label
        sum will be padded to ensure each are equal to the original unpartitioned
        label distribution.
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
    >>>
    >>> num_clients = 1_000
    >>> num_unique_labels_per_client = 2
    >>>
    >>> # Generate a vector from a log-normal probability distribution
    >>> rng = np.random.default_rng(2024)
    >>> mu, sigma = 0., 2.
    >>> distribution_proba = rng.lognormal(mu, sigma, (num_clients*num_unique_labels_per_client,))
    >>>
    >>> partitioner = DistributionPartitioner(
    >>>     distribution_array=distribution_proba,
    >>>     num_partitions=num_clients,
    >>>     num_unique_labels_per_partition=num_labels_per_client,
    >>>     partition_by="label",  # Assumes that the dataset has a target column `label`
    >>>     preassigned_num_samples_per_label=5,
    >>> )
    >>> fds = FederatedDataset(dataset="mnist", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    >>> print(partition[0])  # Print the first example
    ...
    """

    def __init__(
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
        # self._check_num_partitions_correctness_if_needed()
        self._check_distribution_array_shape()
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
            # Will this ever get triggered since partition is recreated every round?
            return

        # Compute the label distribution from the dataset
        # TODO: Possibly refactor so that it's lazily loaded
        df = self.dataset.to_pandas()
        if self._partition_by not in df:
            raise ValueError(
                f"Target label `{self._partition_by}` does not exist in this dataset"
            )
        label_distribution_dict = df[self._partition_by].value_counts().to_dict()
        label_distribution = np.fromiter(label_distribution_dict.values(), dtype=float)

        # Compute the normalized distribution for each class label
        self._distribution_array = self._distribution_array / np.sum(
            self._distribution_array, axis=-1, keepdims=True
        )

        # Compute the total preassigned number of samples per label for all labels
        # and partitions. This sum will be subtracted
        # from the label distribution from the original dataset, and added back later.
        # It ensures that (1) each partition will have at least
        # `self._preassigned_num_samples_per_label` and (2) there is sufficient
        # indices to sample from the dataset.
        total_preassigned_samples = int(
            self._preassigned_num_samples_per_label
            * self._num_unique_labels_per_partition
            * self._num_partitions
            / self._num_unique_labels
        )

        # Subtract the preassigned total amount from the label distribution,
        # we'll add these back later.
        label_distribution -= total_preassigned_samples

        # Rescale normalized distribution with the actual label distribution.
        # Each row represents the number of samples to be taken for that class label
        # and the sum of each row equals the total of each class label.
        # TODO: Skip this step if rescale = False
        label_sampling_matrix = np.floor(
            (self._distribution_array * label_distribution[:, np.newaxis])
        ).astype(int)

        # Add back the preassigned total amount
        label_sampling_matrix += self._preassigned_num_samples_per_label

        # Create the label sampling dictionary
        label_samples = {
            k: v for k, v in zip(label_distribution_dict.keys(), label_sampling_matrix)
        }

        # CREATE PARTITION SAMPLING DICTIONARY
        partition_id_to_samples: Dict[int, Dict[int, int]] = {
            k: {} for k in range(self._num_partitions)
        }

        # Create encoded class label to get the correct labels.
        encoded_class_label = np.asarray(sorted(self.dataset.unique('label')))

        # Initialize sample tracker. Keys are the same as the class labels.
        # Values are the smallest indices of each array in `label_sampling_dict`
        # which will be sampled. Once a sample is taken from a label/key,
        # increment the value (index) by 1.
        tracker_dict = {k: 0 for k, _ in label_samples.items()}

        for partition_id in range(self._num_partitions):
            # Get the `num_unique_labels_per_partition` labels for each client.
            # Use roll to get the correct index labels for the client (for pathological split).
            labels = np.roll(encoded_class_label, -partition_id)[
                : self._num_unique_labels_per_partition
            ]
            for label in labels:
                index_to_sample = tracker_dict[label]
                partition_id_to_samples[partition_id][label] = label_samples[label][
                    index_to_sample
                ]
                # Increment index to the next value in the sample array
                tracker_dict[label] += 1

        # LAST STEP
        # Prepare data structure to store indices assigned to partition ids
        partition_id_to_indices: Dict[int, List[int]] = {}
        for nid in range(self._num_partitions):
            partition_id_to_indices[nid] = []

        # Reference mapping is this:
        # {0: {0: 4570, 1: 1185},
        #  1: {1: 4029, 2: 30},
        #  2: {2: 4494, 3: 5003},
        # ...
        # Loop over unique labels
        # Loop over each client
        for unique_label in self._num_unique_labels:
            for partition_id in self._num_partitions:

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...]) if the shuffle is True
        if self._shuffle:
            for indices in partition_id_to_indices.values():
                # In place shuffling
                self._rng.shuffle(indices)
        self._partition_id_to_indices = partition_id_to_indices
        self._partition_id_to_indices_determined = True

    def _check_distribution_array_shape(self) -> None:
        """Test distribution array shape correctness."""
        # Infer the number of unique labels from the size of the 1st dimension
        # in distribution array
        self._num_unique_labels = np.shape(self._distribution_array)[0]
        expected_num_columns = (
            self._num_unique_labels_per_partition
            * self._num_partitions
            / self._num_unique_labels
        )
        if expected_num_columns != np.shape(self._distribution_array)[1]:
            raise ValueError(
                "The size of the 2nd dimension in the distribution array needs to be "
                "equal to "
                "`num_unique_labels_per_partition*num_partitions/num_unique_labels`."
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
