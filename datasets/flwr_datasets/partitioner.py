"""Partitioner class that works with HuggingFace Dataset."""
from abc import ABC, abstractmethod
from typing import Optional

import datasets
from datasets import Dataset


class Partitioner(ABC):
    """The base partitioner class that enables obtaining federated partitions.

    The initialization is intended to take all necessary arguments such that the call to
    the `load_partition` method can be use the same for all partitioners.
    """

    def __init__(self):
        self._dataset: Optional[Dataset] = None

    @property
    def dataset(self):
        """Dataset property."""
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset):
        if self._dataset is None:
            self._dataset = value
        else:
            raise ValueError(
                "The dataset can be assigned only once to the partitioner."
            )

    @abstractmethod
    def load_partition(self, partition_index: int) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        partition_index: int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition: Dataset
            single dataset partition
        """
        raise NotImplementedError

    def _check_if_dataset_assigned(self):
        """Check if dataset is assigned - it should be prior to using load_partition."""
        if self._dataset is None:
            raise ValueError(
                "The dataset field should be set before using the load_partition."
            )


class IidPartitioner(Partitioner):
    """Partitioner creates each partition sampled randomly from the dataset.

    Parameters
    ----------
    num_partitions: int
        The total number of partitions that the data will be divided into.
    """

    def __init__(self, num_partitions: int) -> None:
        super().__init__()
        self._num_partitions = num_partitions

    def load_partition(self, partition_index: int) -> datasets.Dataset:
        """Load a single iid partition based on the partition index."""
        self._check_if_dataset_assigned()
        return self.dataset.shard(
            num_shards=self._num_partitions, index=partition_index, contiguous=True
        )
