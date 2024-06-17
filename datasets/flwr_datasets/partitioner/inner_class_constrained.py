# # Copyright 2024 Flower Labs GmbH. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================
# """InnerClassConstrained partitioner class."""
import warnings
from typing import Dict, List, Optional, Union

import numpy as np

import datasets
from datasets import Dataset

from flwr_datasets.common.typing import NDArrayFloat, NDArrayInt
from flwr_datasets.partitioner import Partitioner
from flwr_datasets.partitioner.inner_probability_partitioner import \
    InnerProbabilityPartitioner


class InnerClassConstrainedPartitioner(Partitioner):

    def __init__(self, partition_by: str, num_classes_per_partition: int,
                 partition_sizes: Optional[Union[NDArrayInt, List[int]]]):
        super().__init__()
        self._partition_by = partition_by
        self._num_classes_per_partition = num_classes_per_partition
        self._partition_sizes = partition_sizes
        self._inner_prob_partitioner: Optional[InnerProbabilityPartitioner] = None
        self._inner_prob_setup = False

    def load_partition(self, partition_id: int) -> Dataset:
        # create the partition sizes
        self._setup_inner_prob_if_needed()
        return self._inner_prob_partitioner.load_partition(partition_id=partition_id)

    def _setup_inner_prob_if_needed(self):
        if not self._inner_prob_setup:
            num_unique_labels = self.dataset.unique(self._partition_by)

            # todo: some sanity checks
            per_partition_per_label_prob = 1.0 / self._num_classes_per_partition
            prob_per_partition = ([
                                     per_partition_per_label_prob] *
                                  self._num_classes_per_partition + [
                                     0.0] * (
                                             len(num_unique_labels) -
                                             self._num_classes_per_partition))

            num_partitions = len(self._partition_sizes)
            probs = []
            for i in range(num_partitions):
                prob = prob_per_partition.copy()
                np.random.shuffle(prob)
                probs.append(prob)
            self._inner_prob_partitioner = InnerProbabilityPartitioner(
                self._partition_by, probabilities=np.array(probs),
                partition_sizes=self._partition_sizes)
            self._inner_prob_partitioner.dataset = self.dataset
            self._inner_prob_setup = True

    @property
    def num_partitions(self) -> int:
        return self._inner_prob_partitioner.num_partitions
