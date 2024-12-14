# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
# Copyright zk0 DBA. All Rights Reserved.
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
"""Partitioner class that works with the Hugging Face LeRobot Datasets."""

from logging import INFO
from pathlib import Path
from typing import Callable

from flwr_datasets.partitioner.partitioner import Partitioner
from lerobot.common.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    DATA_DIR,
    LeRobotDataset,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    load_episode_data_index,
    load_hf_dataset,
    load_info,
    load_previous_and_future_frames,
    load_stats,
    load_videos,
    reset_episode_index,
)

import datasets
from flwr.common.logger import log


class FilteredLeRobotDataset(LeRobotDataset):
    """
    Delays loading and processing of dataset until load function is called with an optional filter argument.
    """

    def __init__(
        self,
        repo_id: str,
        root: Path | None = DATA_DIR,
        split: str = "train",
        hf_filter_fn: Callable | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        video_backend: str | None = None,
    ):
        self.repo_id = repo_id
        self.root = root
        self.split = split
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.video_backend = video_backend
        self.hf_dataset = load_hf_dataset(
            self.repo_id, CODEBASE_VERSION, self.root, self.split
        )
        if hf_filter_fn is not None:
            self.hf_dataset = self.hf_dataset.filter(function=hf_filter_fn)
            # after filtering, the stored episode data index may not be the same
            # so let's calculate it on the filtered data
            self.episode_data_index = calculate_episode_data_index(self.hf_dataset)
            self.hf_dataset = reset_episode_index(self.hf_dataset)
        else:
            # if the dataset was not filtered, the saved episode data index can save some time and memory
            if self.split == "train":
                self.episode_data_index = load_episode_data_index(
                    self.repo_id, CODEBASE_VERSION, self.root
                )
            else:
                self.episode_data_index = calculate_episode_data_index(self.hf_dataset)
                self.hf_dataset = reset_episode_index(self.hf_dataset)
        self.stats = load_stats(self.repo_id, CODEBASE_VERSION, self.root)
        self.info = load_info(self.repo_id, CODEBASE_VERSION, self.root)
        if self.video:
            self.videos_dir = load_videos(self.repo_id, CODEBASE_VERSION, self.root)
            self.video_backend = (
                self.video_backend if self.video_backend is not None else "pyav"
            )


class LeRobotDatasetPartitioner(Partitioner):
    """Partitioner creates each partition with even number of task samples using episode_index % num_partitions = parition_id.

    Parameters
    ----------
    num_partitions : int
        The total number of partitions that the data will be divided into.

    Examples
    --------
    >>> from flwr_datasets import FederatedDataset
    >>> from flwr_datasets.partitioner import IidPartitioner
    >>>
    >>> partitioner = PushtPartitioner(num_partitions=10)
    >>> fds = FederatedDataset(dataset="lerobot/pusht", partitioners={"train": partitioner})
    >>> partition = fds.load_partition(0)
    """

    def __init__(self, num_partitions: int) -> None:
        super().__init__()
        if num_partitions <= 0:
            raise ValueError("The number of partitions must be greater than zero.")
        self._num_partitions = num_partitions
        self._partition_cache = {}

    @property
    def dataset(self) -> dict:
        """Dataset property."""
        if self._dataset is None:
            raise AttributeError(
                "The dataset field should be set before using it (directly, via "
                "`load_partition` or some other method). "
            )
        return self._dataset

    @dataset.setter
    def dataset(self, value: dict) -> None:
        if self._dataset is not None:
            raise ValueError(
                "The dataset should be assigned only once to the partitioner."
                "This operation might also wipe out the saved references to the "
                "created partitions (in case the partitioning scheme needs to create "
                "the full partitioning also in order to return a single partition)."
            )
        if not isinstance(value, dict):
            raise TypeError(
                f"The dataset object you want to assign to the partitioner should be "
                f"of type `datasets.Dataset` but given {type(value)}."
            )
        self._dataset = value

    def load_partition(self, partition_id: int) -> LeRobotDataset:
        """Load a single partition based on the partition index.

        Parameters
        ----------
        partition_id : int
            the index that corresponds to the requested partition

        Returns
        -------
        dataset_partition : Dataset
            single dataset partition
        """
        partition = self._partition_cache.get(partition_id, None)
        if partition is not None:
            # log(
            #     INFO,
            #     f"Reusing cached partition_id={partition_id}. Summary:\n {partition}",
            # )
            return partition
        else:
            partition = FilteredLeRobotDataset(
                repo_id=self.dataset["dataset_name"],
                delta_timestamps=self.dataset["delta_timestamps"],
                hf_filter_fn=lambda x: x["episode_index"] % self._num_partitions
                == partition_id,
            )
            self._partition_cache[partition_id] = partition
            # log(INFO, f"Loaded partition_id={partition_id}. Summary:\n {partition}")
        return partition

    @property
    def num_partitions(self) -> int:
        """Total number of partitions."""
        return self._num_partitions
