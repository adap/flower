from typing import Optional, Union, List, Dict
import datasets
from abc import ABC, abstractmethod
import numpy as np
from datasets import Dataset, DatasetDict


class Sampler(ABC):
    # does it make a difference if it should work on only the references
    # or also on the data
    def __init__(
            self,
            n_partitions: Optional[int] = None,
            partition_size: Optional[int] = None,
            partition_by: Optional[str] = None,
    ):
        # self.dataset = dataset
        self._n_partitions = n_partitions
        self._partition_size = partition_size
        self._partition_by = partition_by

    def _determine_n_partitions(self, dataset):
        dataset_length = dataset.num_rows
        # The remainder is taken care of, document it
        self._n_partitions = dataset_length // self._partition_size

    @abstractmethod
    def get_partition(self, dataset,
                      partition_index: Union[int, str]) -> datasets.Dataset:
        pass

    @abstractmethod
    def get_partitions(self, dataset) -> List[datasets.Dataset]:
        raise NotImplementedError

    # This methods might be needed for all in samplers in order to provide
    # reproducibility
    # def create_partition_references(
    #         self, indices, labels, sampler
    # ) -> List[List[int]]:  # or map of lists ?
    #
    # def save_partition_references(self):
    #     pass


class CIDSampler(Sampler):  # or Natural Division Sampler
    def __init__(
            self,
            # dataset: datasets.Dataset,
            n_partitions: Optional[int] = None,
            partition_size: Optional[int] = None,
            partition_by: Optional[str] = None
    ):
        super().__init__(n_partitions, partition_size, partition_by)
        self._index_to_cid: Dict[int, str] = {}

    def _create_int_idx_to_cid(self, dataset):
        unique_cid = dataset.unique(self._partition_by)
        index_to_cid = {index: cid for index, cid in
                        zip(range(len(unique_cid)), unique_cid)}
        self._index_to_cid = index_to_cid

    def get_partition(self, dataset, partition_index) -> datasets.Dataset:
        if self._n_partitions is None:
            self._determine_n_partitions(dataset)
        if len(self._index_to_cid) == 0:
            self._create_int_idx_to_cid(dataset)
        if self._n_partitions is None:
            self._n_partitions = len(self._index_to_cid)
        return dataset.filter(
            lambda row: row[self._partition_by] == self._index_to_cid[partition_index])

    def get_partitions(self, dataset) -> List[datasets.Dataset]:
        if self._n_partitions is None:
            self._determine_n_partitions(dataset)
        if len(self._index_to_cid) == 0:
            self._create_int_idx_to_cid(dataset)
        if self._n_partitions is None:
            self._n_partitions = len(self._index_to_cid)
        partitions = []
        for partition_index in range(self._n_partitions):
            partitions.append(self.get_partition(dataset, partition_index))
        return partitions


class IIDSampler(Sampler):
    def __init__(
            self,
            # dataset: datasets.Dataset,
            n_partitions: Optional[int] = None,
            partition_size: Optional[int] = None):
        super().__init__(n_partitions, partition_size, partition_by=None)

    def get_partition(self, dataset, partition_index) -> datasets.Dataset:
        return dataset.shard(num_shards=self._n_partitions, index=partition_index,
                             contiguous=True)

    def get_partitions(self, dataset) -> List[datasets.Dataset]:
        partitions = []
        for partition_index in range(self._n_partitions):
            partitions.append(self.get_partition(dataset, partition_index))
        return partitions


class PowerLawSampler(Sampler):
    # That implementation => probably all implementations need to change
    # The dataset/reference to indices should be kept after the initial get_partition
    # There should be no need to give the datasets each time the get_partition is called
    # At the same time there should not be the need to give the dataset while
    # initialization because it doesn't allow to do the dependency inject with this
    # class. Setter after the initialization in a strange option too. That leaves me
    # with more convoluted creation e.g. using the factory_method with a string name to
    # sampler dictionary and specification of additional keyword. Alternatively (not
    # (I don't like it) there might be a dict-like object - SamplerConfig but it's very
    # similar to the factory method BUT additinally adds the complexity by creating a
    # custom new "class" - the dict

    # Maybe further abstraction is needed to capture the sampling methods for
    # artificially divided. Yet there are some sampler that might be used in case of
    # both, so it might be implicitly assumed that fi the partition_by is None then
    # the dataset should be treated as artificially divided
    def __init__(
            self,
            dataset: datasets.Dataset,
            n_partitions: Optional[int] = None,
            partition_size: Optional[int] = None,
            partition_by: Optional[str] = None,
            min_partition_size: Optional[int] = None,
            n_labels_per_partition=None,
            mean: Optional[float] = 0.0,
            sigma: Optional[float] = 2.0,
    ):
        super().__init__(n_partitions, partition_size, partition_by)
        self._min_partition_size = min_partition_size
        self._n_labels_per_partition = n_labels_per_partition
        self._mean: float = mean
        self._sigma: float = sigma
        self._sorted_dataset: Optional[Dataset] = None
        self._partition_reference: List[List[int]] = self._create_partition_references(
            dataset)

    def get_partition(self, dataset: Dataset, partition_index) -> datasets.Dataset:
        return self._sorted_dataset.select(self._partition_reference[partition_index])

    def get_partitions(self, dataset) -> List[datasets.Dataset]:
        partitions = []
        for partition_index in range(self._n_partitions):
            partitions.append(self.get_partition(dataset, partition_index))
        return partitions

    def _create_partition_references(self, dataset: Dataset):
        self._sorted_dataset = dataset.sort("label")
        labels = self._sorted_dataset["label"]
        full_idx = range(len(labels))

        class_counts = np.bincount(labels)
        labels_cs = np.cumsum(class_counts)
        labels_cs = [0] + labels_cs[:-1].tolist()

        partitions_idx = []
        num_classes = len(np.bincount(labels))
        hist = np.zeros(num_classes, dtype=np.int32)

        # assign min_samples_per_partition
        min_samples_per_label_per_partition = int(
            self._min_partition_size / self._n_labels_per_partition)
        for u_id in range(self._n_partitions):
            partitions_idx.append([])
            for cls_idx in range(self._n_labels_per_partition):
                # label for the u_id-th client
                cls = (u_id + cls_idx) % num_classes
                # record minimum data
                indices = list(
                    full_idx[
                    labels_cs[cls]
                    + hist[cls]: labels_cs[cls]
                                 + hist[cls]
                                 + min_samples_per_label_per_partition
                    ]
                )
                partitions_idx[-1].extend(indices)
                hist[cls] += min_samples_per_label_per_partition

        # add remaining images following power-law
        probs = np.random.lognormal(
            self._mean,
            self._sigma,
            (
                num_classes, int(self._n_partitions / num_classes),
                self._n_labels_per_partition),
        )
        remaining_per_class = class_counts - hist
        # obtain how many samples each partition should be assigned for each of the
        # labels it contains
        probs = (
                remaining_per_class.reshape(-1, 1, 1)
                * probs
                / np.sum(probs, (1, 2), keepdims=True)
        )

        for u_id in range(self._n_partitions):
            for cls_idx in range(self._n_labels_per_partition):
                cls = (u_id + cls_idx) % num_classes
                count = int(probs[cls, u_id // num_classes, cls_idx])

                # add count of specific class to partition
                indices = full_idx[
                          labels_cs[cls] + hist[cls]: labels_cs[cls] + hist[cls] + count
                          ]
                partitions_idx[u_id].extend(indices)
                hist[cls] += count

        return partitions_idx
