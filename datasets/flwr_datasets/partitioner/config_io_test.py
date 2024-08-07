# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
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
"""Test the configuration input/output of all partitioners."""


import os
import tempfile
import unittest

from federated_dataset_test import datasets_are_equal
from parameterized import parameterized_class

from datasets import Dataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets.partitioner.partitioner import Partitioner
from partitioner import LinearPartitioner, ExponentialPartitioner, SquarePartitioner, \
    ShardPartitioner, InnerDirichletPartitioner

partitioner_test_cases = [
    {"partitioner": IidPartitioner, "kwargs": {"num_partitions": 5}},
    {
        "partitioner": DirichletPartitioner,
        "kwargs": {
            "num_partitions": 5,
            "partition_by": "labels",
            "alpha": 0.1,
            "min_partition_size": 0,
        },
    },
    {
        "partitioner": DirichletPartitioner,
        "kwargs": {
            "num_partitions": 5,
            "partition_by": "labels",
            "alpha": [0.1, 0.1, 0.1, 0.1, 0.1],
            "min_partition_size": 0,
        },
    },
    {
        "partitioner": InnerDirichletPartitioner,
        "kwargs": {
            "partition_sizes": [20] * 5,
            "partition_by": "labels",
            "alpha": 0.5,

        },
    },
    {
        "partitioner": LinearPartitioner,
        "kwargs": {
            "num_partitions": 5,
        },
    },
    {
        "partitioner": SquarePartitioner,
        "kwargs": {
            "num_partitions": 5,
        },
    },
    {
        "partitioner": ExponentialPartitioner,
        "kwargs": {
            "num_partitions": 5,
        },
    },
    {
        "partitioner": ShardPartitioner,
        "kwargs": {
            "num_partitions":4,
            "partition_by": "labels",
            "shard_size": 5,
            "num_shards_per_partition": 2,

        },
    },

]


@parameterized_class(partitioner_test_cases)
class TestPartitionerAndConfigsInMemory(unittest.TestCase):
    """Test to_config and from_config which does not involve using on disk files."""

    partitioner: Partitioner

    def setUp(self) -> None:
        """Set up the partitioner from the class parameters."""
        self.partitioner = self.partitioner(**self.kwargs)
        num_rows = 100
        data = {
            "features": list(range(num_rows)),
            "labels": [i % 2 for i in range(num_rows)],
        }
        dataset = Dataset.from_dict(data)
        self.dataset = dataset
        self.partitioner.dataset = dataset

    def test_config_to_and_from_same_first_partition(self) -> None:
        """Test if partitioner is correctly recreated.

        Transform first partitioner to config, then load it back using from_config.
        """
        config = self.partitioner.to_config()
        partition_from_original = self.partitioner.load_partition(0)
        recreated_partitioner = self.partitioner.from_config(config)
        recreated_partitioner.dataset = self.dataset
        partition_from_recreated = recreated_partitioner.load_partition(0)
        self.assertTrue(
            datasets_are_equal(partition_from_original, partition_from_recreated)
        )

    def test_config_to_and_from_same_first_partition_with_indices(self) -> None:
        """Test if the partitioner is correctly recreated with indices passed.

        The _partition_id_to_indices_determined should be marked as True in order to
        avoid the recreation of _partition_id_to_indices.
        """
        config = self.partitioner.to_config()
        indices = self.partitioner.partition_id_to_indices

        recreated_partitioner = self.partitioner.from_config(
            config, partition_id_to_indices=indices
        )
        recreated_partitioner.dataset = self.dataset

        self.assertTrue(
            recreated_partitioner._partition_id_to_indices_determined,
            "The _partition_id_to_indices_determined should be set to True after "
            "from_config",
        )


@parameterized_class(partitioner_test_cases)
class TestPartitionersAndConfigsInFiles(unittest.TestCase):
    """Test to_config_file and from_config_file which involves using on disk files."""

    partitioner: Partitioner

    def setUp(self) -> None:
        """Set up the partitioner from the class parameters."""
        self.partitioner = self.partitioner(**self.kwargs)
        num_rows = 100
        data = {
            "features": list(range(num_rows)),
            "labels": [i % 2 for i in range(num_rows)],
        }
        dataset = Dataset.from_dict(data)
        self.dataset = dataset
        self.partitioner.dataset = dataset

    def test_to_and_from_config_file_no_indices(self) -> None:
        """Test to_config_file and from_config_file methods creates the same object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")

            self.partitioner.to_config_file(
                config_path, include_partition_id_to_indices=False
            )

            recreated_partitioner = self.partitioner.from_config_file(
                config_path, infer_partition_id_to_indices=False
            )
            recreated_partitioner.dataset = self.dataset

            partition_from_original = self.partitioner.load_partition(0)
            partition_from_recreated = recreated_partitioner.load_partition(0)

            self.assertTrue(
                datasets_are_equal(partition_from_original, partition_from_recreated),
                "Partitions should be equal after loading from config.",
            )

    def test_to_and_from_config_file_no_indices_but_try_to_infer(self) -> None:
        """Test to_config_file and from_config_file methods creates the same object."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")

            self.partitioner.to_config_file(
                config_path, include_partition_id_to_indices=False
            )
            with self.assertRaises(FileNotFoundError):
                _ = self.partitioner.from_config_file(
                    config_path, infer_partition_id_to_indices=True
                )

    def test_to_and_from_config_file(self) -> None:
        """Test to_config_file and from_config_file methods produce the same results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            indices_path = os.path.join(temp_dir, "indices.json")

            self.partitioner.to_config_file(
                config_path,
                include_partition_id_to_indices=True,
                indices_path=indices_path,
            )

            recreated_partitioner = self.partitioner.from_config_file(config_path)
            recreated_partitioner.dataset = self.dataset

            partition_from_original = self.partitioner.load_partition(0)
            partition_from_recreated = recreated_partitioner.load_partition(0)

            self.assertTrue(
                datasets_are_equal(partition_from_original, partition_from_recreated),
                "Partitions should be equal after loading from config.",
            )

    def test_to_and_from_config_file_no_indices_path(self) -> None:
        """Test to_config_file and from_config_file without giving indices path.

        The indices path is supposed to be created based on the original name.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            indices_path = None

            self.partitioner.to_config_file(
                config_path,
                include_partition_id_to_indices=True,
                indices_path=indices_path,
            )

            recreated_partitioner = self.partitioner.from_config_file(
                config_path=config_path
            )
            recreated_partitioner.dataset = self.dataset

            partition_from_original = self.partitioner.load_partition(0)
            partition_from_recreated = recreated_partitioner.load_partition(0)

            self.assertTrue(
                datasets_are_equal(partition_from_original, partition_from_recreated),
                "Partitions should be equal after loading from config.",
            )

    def test_to_and_from_config_file_no_indices_path_exist(self) -> None:
        """Test the existence of indices path when not given explicitly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, "config.yaml")
            indices_path = None

            self.partitioner.to_config_file(
                config_path,
                include_partition_id_to_indices=True,
                indices_path=indices_path,
            )

            config_dir_path, config_name = os.path.split(config_path)
            config_name_no_ext, extension = os.path.splitext(config_name)
            expected_indices_path = os.path.join(
                config_dir_path, config_name_no_ext + "_indices" + extension
            )
            self.assertTrue(os.path.exists(expected_indices_path))


if __name__ == "__main__":
    unittest.main()
