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
"""FederatedDataset."""


from typing import Callable, Dict, Optional, Tuple, Union
import datasets
from datasets import Dataset, DatasetDict
from flwr_datasets.merge_splitter import MergeSplitter
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.utils import _check_if_dataset_tested, _instantiate_partitioners

Resplitter = Callable[[DatasetDict], DatasetDict]


class FederatedDataset:
    """Representation of a dataset for federated learning/evaluation/analytics.

     Download, partition data among clients (edge devices), or load full dataset.

     Partitions are created using IidPartitioner. Support for different partitioners
     specification and types will come in future releases.

    Parameters
    ----------
    dataset: str
        The name of the dataset in the Hugging Face Hub.
    resplitter: Optional[Union[Resplitter, Dict[Tuple[str, ...], str]]]
        Resplit strategy or custom Callable that transforms the dataset.
    partitioners: Dict[str, Union[Partitioner, int]]
        A dictionary mapping the Dataset split (a `str`) to a `Partitioner` or an `int`
        (representing the number of IID partitions that this split should be partitioned
         into).

    Examples
    --------
    Use MNIST dataset for Federated Learning with 100 clients (edge devices):

    >>> mnist_fds = FederatedDataset(dataset="mnist", partitioners={"train": 100})

    Load partition for client with ID 10.

    >>> partition = mnist_fds.load_partition(10, "train")

    Use test split for centralized evaluation.

    >>> centralized = mnist_fds.load_full("test")
    """

    def __init__(
        self,
        *,
        dataset: str,
        resplitter: Optional[Union[Resplitter, Dict[Tuple[str, ...], str]]] = None,
        partitioners: Dict[str, Union[Partitioner, int]],
    ) -> None:
        _check_if_dataset_tested(dataset)
        self._dataset_name: str = dataset
        self._resplitter = resplitter
        self._partitioners: Dict[str, Partitioner] = _instantiate_partitioners(
            partitioners
        )
        #  Init (download) lazily on the first call to `load_partition` or `load_full`
        self._dataset: Optional[DatasetDict] = None
        self._resplit: bool = False  # Indicate if the resplit happened

    def load_partition(self, idx: int, split: str) -> Dataset:
        """Load the partition specified by the idx in the selected split.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_full` is made.

        Parameters
        ----------
        idx: int
            Partition index for the selected split, idx in {0, ..., num_partitions - 1}.
        split: str
            Name of the (partitioned) split (e.g. "train", "test").

        Returns
        -------
        partition: Dataset
            Single partition from the dataset split.
        """
        self._download_dataset_if_none()
        self._resplit_dataset_if_needed()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        self._check_if_split_present(split)
        self._check_if_split_possible_to_federate(split)
        partitioner: Partitioner = self._partitioners[split]
        self._assign_dataset_to_partitioner(split)
        return partitioner.load_partition(idx)

    def load_full(self, split: str) -> Dataset:
        """Load the full split of the dataset.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_full` is made.

        Parameters
        ----------
        split: str
            Split name of the downloaded dataset (e.g. "train", "test").

        Returns
        -------
        dataset_split: Dataset
            Part of the dataset identified by its split name.
        """
        self._download_dataset_if_none()
        self._resplit_dataset_if_needed()
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        self._check_if_split_present(split)
        return self._dataset[split]

    def _download_dataset_if_none(self) -> None:
        """Lazily load (and potentially download) the Dataset instance into memory."""
        if self._dataset is None:
            self._dataset = datasets.load_dataset(self._dataset_name)

    def _check_if_split_present(self, split: str) -> None:
        """Check if the split (for partitioning or full return) is in the dataset."""
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        available_splits = list(self._dataset.keys())
        if split not in available_splits:
            raise ValueError(
                f"The given split: '{split}' is not present in the dataset's splits: "
                f"'{available_splits}'."
            )

    def _check_if_split_possible_to_federate(self, split: str) -> None:
        """Check if the split has corresponding partitioner."""
        partitioners_keys = list(self._partitioners.keys())
        if split not in partitioners_keys:
            raise ValueError(
                f"The given split: '{split}' does not have a partitioner to perform "
                f"partitioning. Partitioners were specified for the following splits:"
                f"'{partitioners_keys}'."
            )

    def _assign_dataset_to_partitioner(self, split: str) -> None:
        """Assign the corresponding split of the dataset to the partitioner.

        Assign only if the dataset is not assigned yet.
        """
        if self._dataset is None:
            raise ValueError("Dataset is not loaded yet.")
        if not self._partitioners[split].is_dataset_assigned():
            self._partitioners[split].dataset = self._dataset[split]

    def _resplit_dataset_if_needed(self) -> None:
        # this can't be called many times
        # either a new attribute is needed e.g. resplit_dataset
        # or a bool flag that the resplit happened

        # Resplit only once
        if self._resplit:
            return
        if self._dataset is None:
            raise ValueError("The dataset resplit should happen after the download.")
        if self._resplitter:
            resplitter: Resplitter
            if isinstance(self._resplitter, Dict):
                resplitter = MergeSplitter(resplit_strategy=self._resplitter)
            else:
                resplitter = self._resplitter
            self._dataset = resplitter(self._dataset)
        self._resplit = True
