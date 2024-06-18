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
"""Test SemanticPartitioner."""


# pylint: disable=W0212
import unittest
from typing import Tuple

import numpy as np
from datasets import Dataset
from torchvision import models
from parameterized import parameterized

from flwr_datasets.partitioner.semantic_partitioner import SemanticPartitioner


def _dummy_setup(
    num_partitions: int = 3,
    num_rows: int = 10,
    partition_by: str = "label",
    efficient_net_type: int = 0,
    pca_components: int = 128,
    gmm_max_iter: int = 2,
    gmm_init_params: str = "random",
) -> Tuple[Dataset, SemanticPartitioner]:
    """Create a dummy dataset and partitioner for testing."""
    data = {
        "labels": [i % 3 for i in range(num_rows)],
        "features": [np.random.randn(1, 28, 28) for _ in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    partitioner = SemanticPartitioner(
        num_partitions=num_partitions,
        partition_by=partition_by,
        efficient_net_type=efficient_net_type,
        pca_components=pca_components,
        gmm_max_iter=gmm_max_iter,
        gmm_init_params=gmm_init_params,
    )
    partitioner.dataset = dataset
    return dataset, partitioner


class TestSemanticPartitionerSuccess(unittest.TestCase):
    """Test SemanticPartitioner used with no exceptions."""

    @parameterized.expand(  # type: ignore
        [
            # num_partitions, num_rows, partition_by, efficient_net_type, pca_components, gmm_max_iter, gmm_init_way
            (3, 50, "labels", 0, 128, 2, "kmeans"),
            (5, 100, "labels", 0, 256, 1, "random"),
        ]
    )
    def test_valid_initialization(
        self,
        num_partitions: int,
        num_rows: int,
        partition_by: str,
        efficient_net_type: int,
        pca_components: int,
        gmm_max_iter: int,
        gmm_init_way: str,
    ) -> None:
        """Test if alpha is correct scaled based on the given num_partitions."""
        _, partitioner = _dummy_setup(
            num_partitions,
            num_rows,
            partition_by,
            efficient_net_type,
            pca_components,
            gmm_max_iter,
            gmm_init_way,
        )
        self.assertEqual(
            (
                partitioner._num_partitions,
                partitioner._partition_by,
                partitioner._efficient_net_type,
                partitioner._pca_components,
                partitioner._gmm_max_iter,
                partitioner._gmm_init_params,
            ),
            (
                num_partitions,
                partition_by,
                efficient_net_type,
                pca_components,
                gmm_max_iter,
                gmm_init_way,
            ),
        )

    @parameterized.expand([(0,), (1,), (2,), (3,)])
    def test_efficient_net_config(self, efficient_net_type: int):
        """Test if efficient_net_backbone and efficient_net_pretrained_weight are correct."""
        num_partitions, num_rows, partition_by = 3, 100, "labels"
        efficient_nets_dict = [
            (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
            (models.efficientnet_b1, models.EfficientNet_B1_Weights.DEFAULT),
            (models.efficientnet_b2, models.EfficientNet_B2_Weights.DEFAULT),
            (models.efficientnet_b3, models.EfficientNet_B3_Weights.DEFAULT),
            (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            (models.efficientnet_b5, models.EfficientNet_B5_Weights.DEFAULT),
            (models.efficientnet_b6, models.EfficientNet_B6_Weights.DEFAULT),
            (models.efficientnet_b7, models.EfficientNet_B7_Weights.DEFAULT),
        ]
        _, partitioner = _dummy_setup(
            num_partitions,
            num_rows,
            partition_by,
            efficient_net_type=efficient_net_type,
            pca_components=128,
            gmm_max_iter=2,
            gmm_init_params="random",
        )
        self.assertEqual(
            (
                partitioner._efficient_net_backbone,
                partitioner._efficient_net_pretrained_weight,
            ),
            (
                efficient_nets_dict[efficient_net_type][0],
                efficient_nets_dict[efficient_net_type][1],
            ),
        )

    @parameterized.expand([(64,), (128,), (256,)])
    def test_pca_components(self, pca_components: int) -> None:
        """Test if pca_components is correct scaled based on the given num_partitions."""
        num_partitions, num_rows, partition_by = 3, 300, "labels"
        _, partitioner = _dummy_setup(
            num_partitions,
            num_rows,
            partition_by,
            efficient_net_type=0,
            pca_components=pca_components,
            gmm_max_iter=2,
            gmm_init_params="random",
        )
        self.assertEqual(partitioner._pca_components, pca_components)

    @parameterized.expand([(2, "random"), (2, "kmeans"), (2, "k-means++")])
    def test_gaussian_mixture_model(
        self, gmm_max_iter: int, gmm_init_params: str
    ) -> None:
        """Test if gmm_max_iter and gmm_init_way are correct."""
        num_partitions, num_rows, partition_by = 3, 300, "labels"
        _, partitioner = _dummy_setup(
            num_partitions,
            num_rows,
            partition_by,
            efficient_net_type=0,
            pca_components=128,
            gmm_max_iter=gmm_max_iter,
            gmm_init_params=gmm_init_params,
        )
        self.assertEqual(
            (partitioner._gmm_max_iter, partitioner._gmm_init_params),
            (gmm_max_iter, gmm_init_params),
        )

    def test_determine_partition_id_to_indices(self) -> None:
        """Test the determine_nod_id_to_indices matches the flag after the call."""
        num_partitions, num_rows, partition_by = 3, 300, "labels"
        _, partitioner = _dummy_setup(num_partitions, num_rows, partition_by)
        partitioner._determine_partition_id_to_indices_if_needed()
        self.assertTrue(
            partitioner._partition_id_to_indices_determined
            and len(partitioner._partition_id_to_indices) == num_partitions
        )


class TestSemanticPartitionerFailure(unittest.TestCase):
    """Test SemanticPartitioner failures (exceptions) by incorrect usage."""

    @parameterized.expand([(-2,), (-1,), (3,), (4,), (100,)])  # type: ignore
    def test_load_invalid_partition_index(self, partition_id):
        """Test if raises when the load_partition is above the num_partitions."""
        _, partitioner = _dummy_setup(
            num_partitions=3,
            num_rows=300,
            partition_by="labels",
            efficient_net_type=0,
            gmm_max_iter=2,
            gmm_init_params="random",
        )
        with self.assertRaises(KeyError):
            partitioner.load_partition(partition_id)

    @parameterized.expand([(-1,), (2.5,), (9,), (8), (7.0,)])
    def test_invalid_efficient_net_type(self, efficient_net_type):
        """
        Test if efficient_net_type is not an integer or not in range [0, 7]."""
        with self.assertRaises((ValueError, TypeError)):
            SemanticPartitioner(
                num_partitions=2,
                efficient_net_type=efficient_net_type,
                partition_by="labels",
                gmm_max_iter=2,
            )

    @parameterized.expand(  # type: ignore
        [(0,), (-1,), (11,), (100,)]
    )  # num_partitions,
    def test_invalid_num_partitions(self, num_partitions):
        """Test if 0 is invalid num_partitions."""
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(
                num_partitions=num_partitions, num_rows=10, partition_by="labels"
            )
            partitioner.load_partition(0)

    @parameterized.expand([(0,), (-1,), (2.0,), (11,)])
    def test_invalid_pca_components(self, pca_components):
        """Test if pca_components is not a positive integer."""
        with self.assertRaises((ValueError, TypeError)):
            _, partitioner = _dummy_setup(
                num_partitions=2,
                num_rows=10,
                partition_by="labels",
                pca_components=pca_components,
            )
            partitioner.load_partition(0)

    @parameterized.expand(
        [
            (0, "random"),
            (-1, "keams"),
            (2.0, "k-means++"),
            (2, "rand"),
            (3, "kmean"),
            (10, "kmeans++"),
        ]
    )
    def test_invalid_gaussian_mixture_config(self, gmm_max_iter, gmm_init_params):
        """
        Test if gmm_max_iter is not a positive integer or gmm_init_params is not one of the allowed values.
        """
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(
                num_partitions=2,
                num_rows=10,
                partition_by="labels",
                gmm_max_iter=gmm_max_iter,
                gmm_init_params=gmm_init_params,
            )
            partitioner.load_partition(0)


if __name__ == "__main__":
    unittest.main()
