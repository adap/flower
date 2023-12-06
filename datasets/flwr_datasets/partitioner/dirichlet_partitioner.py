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
"""Dirichlet partitioner class that works with Hugging Face Datasets."""
from typing import Dict, List, Optional, Union

import numpy as np

import datasets
from flwr_datasets.common.typing import NDArrayFloat
from flwr_datasets.partitioner.partitioner import Partitioner


class DirichletPartitioner(Partitioner):  # pylint: disable=R0902
    """Partitioner based on Dirichlet distribution.

    The balancing (not mentioned in paper but implemented in the code) is controlled by
    `self_balancing` parameter.

    Implementation based on Bayesian Nonparametric Federated Learning of Neural Networks
    https://arxiv.org/abs/1905.12022

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.
    alpha : Union[float, List[float], NDArrayFloat]
        Concentration parameter to the Dirichlet distribution
    partition_by : str
        Column name of the labels (targets) based on which Dirichlet sampling works.
    min_partition_size : int
        The minimum number of samples that each partitions will have (the sampling
        process is repeated if any partition is too small).
    self_balancing: bool
        Weather assign further samples to a partition after the number of samples
        exceeded the average number of samples per partition. (True in the original
        paper's code although not mentioned in paper itself).
    """

    def __init__(  # pylint: disable=R0913
        self,
        num_partitions: int,
        alpha: Union[float, List[float], NDArrayFloat],
        partition_by: str,
        min_partition_size: Optional[int] = None,
        self_balancing: bool = True,
        shuffle: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        # Attributes based on the constructor
        self._num_partitions = num_partitions
        self._alpha: NDArrayFloat = self._initialize_alpha(alpha)
        self._partition_by = partition_by
        if min_partition_size is None:
            # Note that zero might make problems with the training
            min_partition_size = 0
        self._min_partition_size: int = min_partition_size
        self._self_balancing = self_balancing
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed=self._seed)  # NumPy random generator

        # Utility attributes
        # The attributes below are determined during the first call to load_partition
        self._num_unique_classes: Optional[int] = None
        self._avg_num_of_samples_per_node: Optional[float] = None
        self._unique_classes: Optional[Union[List[int], List[str]]] = None
        self._node_id_to_indices: Dict[int, List[int]] = {}
        self._node_id_to_indices_determined = False

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

    def _initialize_alpha(
        self, alpha: Union[float, List[float], NDArrayFloat]
    ) -> NDArrayFloat:
        if isinstance(alpha, float):
            alpha = np.array([alpha], dtype=float).repeat(self._num_partitions)
        elif isinstance(alpha, List):
            if len(alpha) != self._num_partitions:
                raise ValueError(
                    "The alpha parameter needs to be of length of equal to the "
                    "num_partitions."
                )
            alpha = np.asarray(alpha)
        elif isinstance(alpha, np.ndarray):
            # pylint: disable=R1720
            if alpha.ndim == 1 and alpha.shape[0] != self._num_partitions:
                raise ValueError(
                    "The alpha parameter needs to be of length of equal to"
                    "the num_partitions."
                )
            elif alpha.ndim == 2:
                alpha = alpha.flatten()
                if alpha.shape[0] != self._num_partitions:
                    raise ValueError(
                        "The alpha parameter needs to be of length of equal to "
                        "the num_partitions."
                    )
        else:
            raise ValueError("The given alpha format is not supported.")
        return alpha

    def _determine_node_id_to_indices_if_needed(self) -> None:  # pylint: disable=R0914
        """Create an assignment of indices to the partition indices."""
        if not self._node_id_to_indices_determined:
            pass

        # Generate information needed for Dirichlet partitioning
        self._unique_classes = self.dataset.unique(self._partition_by)
        assert self._unique_classes is not None
        self._num_unique_classes = len(self._unique_classes)
        # This is needed only if self._self_balancing is True (the default option)
        self._avg_num_of_samples_per_node = self.dataset.num_rows / self._num_partitions

        # Change targets list data type to numpy
        targets = np.array(self.dataset[self._partition_by])

        # Repeat the sampling procedure based on the Dirichlet distribution until the
        # min_partition_size is reached.
        while True:
            # Prepare data structure to store indices assigned to node ids
            node_id_to_indices: Dict[int, List[int]] = {}
            for nid in range(self._num_partitions):
                node_id_to_indices[nid] = []

            # Iterated over all unique labels (they are not necessarily of type int)
            for k in self._unique_classes:
                # Access all the indices associated with class k
                indices_representing_class_k = np.nonzero(targets == k)[0]
                # Determine division (the fractions) of the data representing class k
                # among the partitions
                class_k_division_proportions = self._rng.dirichlet(self._alpha)
                nid_to_proportion_of_k_samples = {}
                for nid in range(self._num_partitions):
                    nid_to_proportion_of_k_samples[nid] = class_k_division_proportions[
                        nid
                    ]
                # Balancing (not mentioned in the paper but implemented)
                # Do not assign additional samples to the node if it already has more
                # than the average numbers of samples per partition. Note that it might
                # especially affect classes that are later in the order. This is the
                # reason for more sparse division that the alpha might suggest.
                if self._self_balancing:
                    assert self._avg_num_of_samples_per_node is not None
                    for nid in nid_to_proportion_of_k_samples.copy():
                        if (
                            len(node_id_to_indices[nid])
                            > self._avg_num_of_samples_per_node
                        ):
                            nid_to_proportion_of_k_samples[nid] = 0

                    # Normalize the proportions such that they sum up to 1
                    sum_proportions = sum(nid_to_proportion_of_k_samples.values())
                    for nid, prop in nid_to_proportion_of_k_samples.copy().items():
                        nid_to_proportion_of_k_samples[nid] = prop / sum_proportions

                # Determine the split indices
                cumsum_division_fractions = np.cumsum(
                    list(nid_to_proportion_of_k_samples.values())
                )
                cumsum_division_numbers = cumsum_division_fractions * len(
                    indices_representing_class_k
                )
                # [:-1] is because the np.split requires the division indices but the
                # last element represents the sum = total number of samples
                indices_on_which_split = cumsum_division_numbers.astype(int)[:-1]

                split_indices = np.split(
                    indices_representing_class_k, indices_on_which_split
                )

                # Append new indices (coming from class k) to the existing indices
                for nid, indices in node_id_to_indices.items():
                    indices.extend(split_indices[nid].tolist())

            # Determine if the indices assignment meets the min_partition_size
            # If it does not mean the requirement repeat the Dirichlet sampling process
            # Otherwise break the while loop
            min_sample_size_on_client = min(
                len(indices) for indices in node_id_to_indices.values()
            )
            if min_sample_size_on_client >= self._min_partition_size:
                break

        # Shuffle the indices not to have the datasets with targets in sequences like
        # [00000, 11111, ...])
        for indices in node_id_to_indices.values():
            # In place shuffling
            self._rng.shuffle(indices)
        self._node_id_to_indices = node_id_to_indices
        self._node_id_to_indices_determined = True

    def _check_num_partitions_correctness_if_needed(self) -> None:
        if not self._node_id_to_indices_determined:
            if self._num_partitions > self.dataset.num_rows or self._num_partitions < 1:
                raise ValueError(
                    "The number of partitions needs to be greater than zero and smaller"
                    " than the number of samples in the dataset."
                )
