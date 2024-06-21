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
"""Partitioner class that works with Hugging Face Datasets."""


import json
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from flwr_datasets.partitioner.yaml_utils import YamlHandler
import yaml

from datasets import Dataset
from flwr_datasets.partitioner.utils import load_partition_id_to_indices, \
    _extract_private_attributes_from_object, _remove_leading_underscores_from_config


class Partitioner(ABC):
    """The base partitioner class that enables obtaining federated partitions.

    The initialization is intended to take all necessary arguments such that the call to
    the `load_partition` method can be used in the same way for all partitioners.
    """

    def __init__(self) -> None:
        self._dataset: Optional[Dataset] = None
        self._partition_id_to_indices: Dict[int, List[int]] = {}

    @property
    def dataset(self) -> Dataset:
        """Dataset property."""
        if self._dataset is None:
            raise AttributeError(
                "The dataset field should be set before using it (directly, via "
                "`load_partition` or some other method). "
            )
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset) -> None:
        if self._dataset is not None:
            raise Exception(
                "The dataset should be assigned only once to the partitioner."
                "This operation might also wipe out the saved references to the "
                "created partitions (in case the partitioning scheme needs to create "
                "the full partitioning also in order to return a single partition)."
            )
        self._dataset = value

    @abstractmethod
    def load_partition(self, partition_id: int) -> Dataset:
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

    def is_dataset_assigned(self) -> bool:
        """Check if a dataset has been assigned to the partitioner.

        This method returns True if a dataset is already set for the partitioner,
        otherwise, it returns False.

        Returns
        -------
        dataset_assigned : bool
            True if a dataset is assigned, otherwise False.
        """
        return self._dataset is not None

    @property
    @abstractmethod
    def num_partitions(self) -> int:
        """Total number of partitions."""

    @property
    @abstractmethod
    def partition_id_to_indices(self) -> Dict[int, List[int]]:
        """Partition id to indices (the result of partitioning)."""

    # @abstractmethod
    def to_config(self) -> Dict[str, Any]:
        """Create a configuration (a dictionary) representing the partitioner.

        This method is used in `to_config_file`.

        Returns
        -------
        partitioner_representation: Dict[str, Any]
            Parameters representing a partitioner.
        """
        config = _extract_private_attributes_from_object(self)
        config["_target_"] = f"{self.__class__.__module__}.{self.__class__.__name__}.from_config"
        return config

    def to_config_file(
        self,
        config_path: Optional[str] = None,
        include_partition_id_to_indices: bool = True,
        indices_path: Optional[str] = None,
    ) -> None:
        """Save a config representing the partitioner in YAML file.

        It uses `to_config` to create the partitioner representation. If
        `partition_id_to_indices` is True the partition_id_to_indices are saved as
        a separate JSON file with path reference in the original configuration file.

        Parameters
        ----------
        config_path: str
            Path where the configuration file will be saved.
        include_partition_id_to_indices: bool
            Whether to save the mapping of partition_id to indices that was created
            when partitioning the dataset.
        indices_path: Optional[str]
            Path where `partition_id_to_indices` will be saved. It has no effect if
            `include_partition_id_to_indices` is False.
        """
        config = self.to_config()
        if include_partition_id_to_indices:
            # Save partition_id_to_indices and store the path in the config
            indices = self.partition_id_to_indices
            if indices_path is None:
                config_dir_path, config_name = os.path.split(config_path)
                config_name_no_ext, extension = os.path.splitext(config_name)
                indices_path = os.path.join(
                    config_dir_path, config_name_no_ext + "_indices" + extension
                )
            config["partition_id_to_indices_path"] = str(indices_path)
            with open(indices_path, "w") as indices_file:
                json.dump(indices, indices_file)

        yaml_handler = YamlHandler()
        yaml_handler.dump(config, config_path)

    @classmethod
    def from_config(
            cls,
            config: Dict[str, Any],
            partition_id_to_indices: Optional[Dict[int, List[int]]] = None,
    ) -> "Partitioner":
        """Instantiate the partitioner based on the given parameters.

        This method should make sure that the object internal state is correctly
        reflected (when indices mapping is inferred there should be no need for new
        partitioning).

        Parameters
        ----------
        config: Dict[str, Any]
            Representation of the partitioner that
        partition_id_to_indices: Optional[Dict[int, List[int]]]
            Mapping of partition_id to indices that was created when partitioning the
            dataset.

        Returns
        -------
        partitioner: Partitioner
            An instantiated partitioner
        """
        config = _remove_leading_underscores_from_config(config)
        partitioner = cls(**config)
        if partition_id_to_indices is not None:
            partitioner._partition_id_to_indices = partition_id_to_indices
            # _partition_id_to_indices_determined needs to be True to avoid
            # re-creation of the indices
            partitioner._partition_id_to_indices_determined = True
        return partitioner

    @classmethod
    def from_config_file(
        cls,
        config_path: str,
        infer_partition_id_to_indices: bool = True,
    ) -> "Partitioner":
        """Instantiate partitioner based on the saved configuration file.

        Parameters
        ----------
        config_path : str
            Path to the configuration file.
        infer_partition_id_to_indices :  True
            Whether to infer partition id to indices mapping that resulted from the
            partitioning.

        Returns
        -------
        partitioner : Partitioner
            The instantiated partitioner based on the provided config.
        """
        yaml_handler = YamlHandler()
        config = yaml_handler.load(config_path)
        partition_id_to_indices = None
        if infer_partition_id_to_indices:
            partition_id_to_indices_path = config.pop(
                "partition_id_to_indices_path", None
            )

            if partition_id_to_indices_path is None:
                warnings.warn(
                    "There are no `partition_id_to_indices_path` value "
                    "in the configuration. The inference of the indices "
                    "path is not possible.",
                    stacklevel=1,
                )
                config_dir_path, config_name = os.path.split(config_path)
                config_name_no_ext, extension = os.path.splitext(config_name)
                partition_id_to_indices_path = os.path.join(
                    config_dir_path, config_name_no_ext + "_indices" + extension
                )
            partition_id_to_indices = load_partition_id_to_indices(
                partition_id_to_indices_path
            )
        partitioner = cls.from_config(config, partition_id_to_indices)
        return partitioner
