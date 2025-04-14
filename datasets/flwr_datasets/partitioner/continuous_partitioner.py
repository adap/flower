# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Continuous partitioner class that works with Hugging Face Datasets."""


from typing import Optional

import numpy as np
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner


class ContinuousPartitioner(Partitioner):
    """Partitioner based on a continuous dataset property with adjustable strictness.

    This partitioner enables non-IID partitioning by sorting the dataset based on a
    continuous property and introducing Gaussian noise controlled by a strictness parameter.

    Parameters
    ----------
    num_partitions : int
        Number of partitions to create.
    partition_by : str
        Name of the continuous feature to partition the dataset on.
    strictness : float
        Controls how strongly the feature influences partitioning (0 = iid, 1 = non-iid).
    shuffle : bool
        Whether to shuffle the indices within each partition (default: True).
    seed : Optional[int]
        Random seed for reproducibility.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import ContinuousPartitioner
    >>>
    >>> partitioner = ContinuousPartitioner(
    >>>     num_partitions=5,
    >>>     partition_by="logS",
    >>>     strictness=0.7,
    >>>     shuffle=True
    >>> )
    >>> fds = FederatedDataset(dataset="chembl_aqsol", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    >>> print(partition[0])
    """

    def __init__(
        self,
        num_partitions: int,
        partition_by: str,
        strictness: float,
        shuffle: bool = True,
        seed: Optional[int] = 42,
    ) -> None:
        super().__init__()
        if not (0 <= strictness <= 1):
            raise ValueError("`strictness` must be between 0 and 1")
        if num_partitions <= 0:
            raise ValueError("`num_partitions` must be greater than 0")

        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._strictness = strictness
        self._shuffle = shuffle
        self._seed = seed
        self._rng = np.random.default_rng(seed)

        # Lazy initialization
        self._partition_id_to_indices: dict[int, list[int]] = {}
        self._partition_id_to_indices_determined = False

    def load_partition(self, partition_id: int) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            The index that corresponds to the requested partition.

        Returns
        -------
        dataset_partition : Dataset
            A single dataset partition.
        """
        self._check_and_generate_partitions_if_needed()
        return self.dataset.select(self._partition_id_to_indices[partition_id])

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        self._check_and_generate_partitions_if_needed()
        return self._num_partitions

    @property
    def partition_id_to_indices(self) -> dict[int, list[int]]:
        """Mapping from partition ID to dataset indices."""
        self._check_and_generate_partitions_if_needed()
        return self._partition_id_to_indices

    def _check_and_generate_partitions_if_needed(self) -> None:
        """Lazy evaluation of the partitioning logic."""
        if self._partition_id_to_indices_determined:
            return

        if self._num_partitions > self.dataset.num_rows:
            raise ValueError(
                "Number of partitions must be less than or equal to number of dataset samples."
            )

        # Extract property values
        property_values = np.array(self.dataset[self._partition_by], dtype=np.float32)

        # Check for missing values (None or NaN)
        if np.any(property_values == None) or np.isnan(property_values).any():
            raise ValueError(
                f"The column '{self._partition_by}' contains None or NaN values, "
                "which are not supported by ContinuousPartitioner. "
                "Please clean or filter your dataset before partitioning."
            )

        # Standardize
        std = np.std(property_values)
        if std < 1e-6:
            raise ValueError(
                f"Cannot standardize column '{self._partition_by}' because it has near-zero standard deviation "
                f"(std={std}). All values are nearly identical, which prevents meaningful partitioning."
            )

        standardized_values = (property_values - np.mean(property_values)) / std

        # Blend noise
        noise = self._rng.normal(loc=0, scale=1, size=len(standardized_values))
        blended_values = (
            self._strictness * standardized_values + (1 - self._strictness) * noise
        )

        # Sort and partition
        sorted_indices = np.argsort(blended_values)
        partition_indices = np.array_split(sorted_indices, self._num_partitions)

        # Create dictionary
        self._partition_id_to_indices = {}
        for pid, indices in enumerate(partition_indices):
            indices = indices.tolist()
            if self._shuffle:
                self._rng.shuffle(indices)
            self._partition_id_to_indices[pid] = indices

        self._partition_id_to_indices_determined = True
