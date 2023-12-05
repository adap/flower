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
from typing import List, Optional, Union

import numpy as np
from common.typing import NDArrayFloat
from partitioner import Partitioner

import datasets


class DirichletPartitioner(Partitioner):
    """Partitioner based on Dirichlet distribution with balancing (not mentioned in
    paper).

    Implementation based on todo:paste-the-url
    """

    def __init__(
        self,
        num_partitions: int,
        alpha: Union[float, List[float], NDArrayFloat],
        partition_by: str,
        min_require_size: Optional[int] = None,
        self_balancing: bool = True,
    ):
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_by = partition_by

        self._dataset_partioned = False
        self._self_balancing = self_balancing

        self._num_unique_classes: Optional[int] = None
        self._alpha: Optional[
            NDArrayFloat
        ] = alpha  # was Non before, check what is right
        # todo: think if that is reasonable (other ppl say # of unique classes)
        if min_require_size is None:
            min_require_size = 0
        self._min_require_size: int = min_require_size

        # todo: some check of the num_partitions

    def load_partition(self, node_id: int) -> datasets.Dataset:
        """Load a partition based on the partition index.

        Parameters
        ----------
        node_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        # If not divided yet, you need to divide the samples
        if not self._dataset_partioned:
            # Determine the number of unique classes
            self._determine_basic_info()

            # This will be modified at the end of the loop to check if the condition
            # of min_required_size is met. If not the next dirichlet sampling will be
            # performed
            min_partition_size_obtained_in_interation = 0

            # maybe labels not targets?
            targets = np.array(self.dataset[self._partition_by])
            while True:
                dataset_indices = list(range(len(self.dataset)))
                # Prepare data structure to store nid_to_indices
                nid_to_indices = {}
                for nid in range(self._num_partitions):
                    nid_to_indices[nid] = []

                # Iterated over all unique labels

                # Not names (int counterpart is needed)
                for k in self._unique_classes:
                    # k is the value of class
                    indices_representing_class_k = np.where(targets == k)[0]
                    class_k_division_proportion = np.random.dirichlet(
                        self._alpha,
                    )
                    # each node gets assigned a proportion of examples that will
                    # get assigned based on the dirichlet probability
                    nid_to_proportion_of_k_samples = {}
                    for nid in range(self._num_partitions):
                        nid_to_proportion_of_k_samples[
                            nid
                        ] = class_k_division_proportion[nid]

                    # Balancing (not mentioned in the paper)
                    if self._self_balancing:
                        for nid, proportion in nid_to_proportion_of_k_samples.items():
                            # if from the previous classes you got too much samples
                            # don't add any further
                            if (
                                len(nid_to_indices[nid])
                                > self._avg_num_of_samples_per_node
                            ):
                                nid_to_proportion_of_k_samples[nid] = 0

                        # Renormalize such that p sums to 1
                        # (apply this opperation even if htere was no change; it
                        # won't modify the values)
                        sum_proportions = nid_to_proportion_of_k_samples.values().sum()

                        for nid, proportion in nid_to_proportion_of_k_samples.items():
                            nid_to_proportion_of_k_samples[nid] = (
                                proportion / sum_proportions
                            )

                    # Determine the split indices
                    # Drop the last one (it adds up to the total number of samples (
                    # todo: check if it does)
                    indices_on_which_split = (
                        np.cumsum(list(nid_to_proportion_of_k_samples.values()))
                        * len(indices_representing_class_k)
                    ).astype(int)[:-1]

                    split_indices = np.split(
                        indices_representing_class_k, indices_on_which_split
                    )

                    # Append to the exisiting indices assignements
                    for nid in nid_to_indices:
                        nid_to_indices[nid].extend(split_indices[nid].tolist())

                # Determine if the assignements meet hte min_samples_size_per_node
                # requriement
                # If not repeat the process
                # Otherwise break the while infinite loop
                min_sample_size_on_client = min(
                    len(indices) for indices in nid_to_indices.values()
                )
                if min_sample_size_on_client >= self._min_require_size:
                    break

            # Shuffling the indices not to have the samples per class in sequences [
            # 00000, 11111 etc)
            self._node_id_to_indices = nid_to_indices
            self._dataset_partioned = True
            return self.dataset.select(self._node_id_to_indices[node_id])
        else:
            return self.dataset.select(self._node_id_to_indices[node_id])

    def _determine_basic_info(self):
        # Determine the basic information that are needed for dirichlet partiioner to
        # start
        self._unique_classes = self.dataset.unique(self._partition_by)

        self._num_unique_classes = len(self._unique_classes)
        self._avg_num_of_samples_per_node = self.dataset.num_rows / self._num_partitions

        if isinstance(self._alpha, float):
            self._alpha = np.array([self._alpha], dtype=float).repeat(
                self._num_partitions
            )
        elif isinstance(self._alpha, List):
            # todo check the correct shape
            self.alpha = np.asarray(self._alpha)
        elif isinstance(self._alpha, NDArrayFloat):
            # todo also check the correct shape
            pass
        else:
            raise ValueError("The alpha type does not match the required one")


if __name__ == "__main__":
    print("hello")
    from datasets import Dataset

    num_rows = 100
    n_unique_natural_ids = 3
    data = {
        "features": list(range(num_rows)),
        "id": [f"{i % n_unique_natural_ids}" for i in range(num_rows)],
        "labels": [i % 2 for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    print(dataset)
    d = DirichletPartitioner(
        num_partitions=10,
        alpha=0.5,
        partition_by="id",
        min_require_size=0,
        self_balancing=False,
    )
    d.dataset = dataset
    p = d.load_partition(0)
    print(p[:])
