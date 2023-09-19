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
"""IID partitioner class that works with Hugging Face Datasets."""


import datasets
from flwr_datasets.partitioner.partitioner import Partitioner


class IidPartitioner(Partitioner):
    """Partitioner creates each partition sampled randomly from the dataset.

    The creation of the indices varies based on the `contiguous` parameter. If it is
    True then e.g. if dataset length is 8 and there are 2 partitions we will
    have the following indices: 1-st {0, 1, 2, 3}, 2-nd partition {4, 5, 6, 7}.
    If it is False then the partitions' indices are determined using np.arange(
    partition_idx, dataset_length, num_partitions) that will produce the following
    indices: 1-st {0, 2, 4, 6}, 2-nd {1, 3, 5, 7}. Check out np.arange documentation
    for more information.

    Parameters
    ----------
    num_partitions: int
        The total number of partitions that the data will be divided into.
    contiguous: bool
        Whether each partition should be created using contiguous indices.
    """

    def __init__(self, num_partitions: int, contiguous: bool = False) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions
        self._contiguous = contiguous

    def load_partition(self, idx: int) -> datasets.Dataset:
        """Load a single IID partition based on the partition index."""
        return self.dataset.shard(
            num_shards=self._num_partitions, index=idx, contiguous=self._contiguous
        )
