"""Federated Dataset."""
from typing import Dict, Optional

from partitioner import IidPartitioner, Partitioner

import datasets
from datasets import Dataset, DatasetDict


class FederatedDataset:
    def __init__(self, *, dataset: str, partitioners: Dict[str, int]) -> None:
        """Representation of a dataset for the federated learning/analytics.

         Download, partition data among clients (edge devices), load full dataset.

         Partitions are created using IidPartitioner. Support for different partitioners
         specification and types will come in the future releases.

        Parameters
        ----------
        dataset: str
            The name of the dataset in the HuggingFace Hub.
        partitioners: Dict[str, int]
            Dataset split to the number of iid partitions.

        """
        self._check_if_dataset_supported(dataset)
        self._dataset_name: str = dataset
        self._partitioners: Dict[str, Partitioner] = self._instantiate_partitioners(
            partitioners
        )

        #  Init (download) lazily on the first call to `load_partition` or `load_full`
        self._dataset: Optional[DatasetDict] = None

    def load_partition(self, idx: int, split: str) -> Dataset:
        """Load the partition specified by the idx in the selected split.

        Parameters
        ----------
        idx: int
            Partition index for the selected split, idx in {0, ..., num_partitions - 1}.
        split: str
            Split name of the dataset to partition (e.g. "train", "test").

        Returns
        -------
        partition: Dataset
            Single partition from the dataset split.
        """
        self._download_dataset_if_none()
        self._check_if_split_present(split)
        self._check_if_split_possible_to_federate(split)

        partitioner = self._partitioners[split]
        return partitioner.load_partition(self._dataset[split], idx)

    def load_full(self, split: str) -> Dataset:
        """Load the full split of the dataset.

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
        return self._dataset[split]

    def _instantiate_partitioners(
        self, partitioners: Dict[str, int]
    ) -> Dict[str, Partitioner]:
        """Transform the partitioners from the initial format to instantiated objects.

        Parameters
        ----------
        partitioners: Dict[str, int]
            Partitioners specified as split to the number of partitions format.

        Returns
        -------
        partitioners: Dict[str, Partitioner]
            Partitioners specified as split to Partitioner object.
        """
        instantiated_partitioners = {}
        for key, value in partitioners.items():
            instantiated_partitioners[key] = IidPartitioner(num_partitions=value)
        return instantiated_partitioners

    def _download_dataset_if_none(self) -> None:
        """Download dataset if the dataset is None = not downloaded yet.

        The dataset is downloaded only when the first call to `load_partition` or
        `load_full` is made.
        """
        if self._dataset is None:
            self._dataset = datasets.load_dataset(self._dataset_name)

    def _check_if_dataset_supported(self, dataset):
        """Check if the dataset is in the narrowed down list of the tested datasets."""
        if dataset not in ["mnist", "cifar10"]:
            raise ValueError(
                f"The currently tested and supported dataset are 'mnist' and "
                f"'cifar10'. Given: {dataset}"
            )

    def _check_if_split_present(self, split: str) -> None:
        """Check if the split (for partitioning or full return) is in the dataset."""
        available_splits = list(self._dataset.keys())
        if split not in available_splits:
            raise ValueError(
                f"The given split: '{split}' is not present in the dataset's splits: "
                f"'{available_splits}'."
            )

    def _check_if_split_possible_to_federate(self, split: str):
        """Check if the split has corresponding partitioner."""
        partitioners_keys = list(self._partitioners.keys())
        if split not in partitioners_keys:
            raise ValueError(
                f"The given split: '{split}' does not have partitioner to perform a "
                f"splits. Partitioners are present for the following splits:"
                f"'{partitioners_keys}'."
            )
