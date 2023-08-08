"""Partitioner class that works with HuggingFace Dataset."""
from abc import ABC, abstractmethod
from typing import Union

import datasets
from datasets import Dataset


class Partitioner(ABC):
    """The base partitioner class that enables obtaining federated partitions.

    The initialization is intended to take all necessary arguments such that the call to
    the `load_partition` method can be use the same for all partitioners.
    """

    @abstractmethod
    def load_partition(
        self, dataset: Dataset, partition_index: Union[int, str]
    ) -> Dataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        dataset: Dataset
            dataset that will be partitioned
        partition_index: Union[int, str]
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition: Dataset
            single dataset partition
        """
        raise NotImplementedError


class IidPartitioner(Partitioner):
    """Partitioner creates each partition sampled randomly from the dataset.

    Parameters
    ----------
    num_partitions: int
        The total number of partitions that the data will be divided into.
    """

    def __init__(self, num_partitions: int):
        super().__init__()
        self._num_partitions = num_partitions

    def load_partition(self, dataset, partition_index) -> datasets.Dataset:
        """Load a single iid partition based on the partition index."""
        return dataset.shard(
            num_shards=self._num_partitions, index=partition_index, contiguous=True
        )
