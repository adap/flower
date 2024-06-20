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
import string
import unittest
from typing import Tuple

import numpy as np
from parameterized import parameterized
from torchvision import models

from datasets import Dataset
from flwr_datasets.partitioner.semantic_partitioner import SemanticPartitioner


def _dummy_setup(
    data_shape: tuple = (28, 28, 1),
    num_partitions: int = 3,
    num_rows: int = 10,
    partition_by: str = "label",
    efficient_net_type: int = 0,
    pca_components: int = 6,
    gmm_max_iter: int = 2,
    gmm_init_params: str = "random",
) -> Tuple[Dataset, SemanticPartitioner]:
    """Create a dummy dataset and partitioner for testing."""
    data = {
        "image": [np.random.randn(*data_shape) for _ in range(num_rows)],
        "label": [i % 3 for i in range(num_rows)],
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

    @parameterized.expand(
        [
            # data_shape, num_partitions, num_rows, partition_by, efficient_net_type, pca_components, gmm_max_iter, gmm_init_params
            ((28, 28, 1), 3, 50, "label", 0, 128, 2, "kmeans"),
            ((1, 28, 28), 5, 100, "label", 2, 256, 1, "random"),
            ((32, 32, 3), 5, 100, "label", 7, 256, 1, "k-means++"),
        ]
    )
    def test_valid_initialization(
        self,
        data_shape: tuple,
        num_partitions: int,
        num_rows: int,
        partition_by: str,
        efficient_net_type: int,
        pca_components: int,
        gmm_max_iter: int,
        gmm_init_params: str,
    ) -> None:
        """Test if alpha is correct scaled based on the given num_partitions."""
        _, partitioner = _dummy_setup(
            data_shape=data_shape,
            num_partitions=num_partitions,
            num_rows=num_rows,
            partition_by=partition_by,
            efficient_net_type=efficient_net_type,
            pca_components=pca_components,
            gmm_max_iter=gmm_max_iter,
            gmm_init_params=gmm_init_params,
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
                gmm_init_params,
            ),
        )

    @parameterized.expand([((28, 28, 1),), ((3, 32, 32),), ((28, 28),)])
    def test_data_shape(self, data_shape: tuple):
        """Test if data_shape is correct."""
        _, partitioner = _dummy_setup(data_shape=data_shape)
        partitioner.load_partition(0)

    @parameterized.expand([(0,), (1,), (2,), (3,)])
    def test_efficient_net_config(self, efficient_net_type: int):
        """Test if efficient_net_backbone and efficient_net_pretrained_weight are correct."""
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
        _, partitioner = _dummy_setup(efficient_net_type=efficient_net_type)
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

    @parameterized.expand([(64,), (96,), (32,)])
    def test_pca_components(self, pca_components: int) -> None:
        """Test if pca_components is correct scaled based on the given num_partitions."""
        _, partitioner = _dummy_setup(num_rows=100, pca_components=pca_components)
        self.assertEqual(partitioner._pca_components, pca_components)

    @parameterized.expand([(1, "random"), (2, "kmeans"), (2, "k-means++")])
    def test_gaussian_mixture_model(
        self, gmm_max_iter: int, gmm_init_params: str
    ) -> None:
        """Test if gmm_max_iter and gmm_init_params are correct."""
        _, partitioner = _dummy_setup(
            gmm_max_iter=gmm_max_iter, gmm_init_params=gmm_init_params
        )
        self.assertEqual(
            (partitioner._gmm_max_iter, partitioner._gmm_init_params),
            (gmm_max_iter, gmm_init_params),
        )

    def test_determine_partition_id_to_indices(self) -> None:
        """Test the determine_nod_id_to_indices matches the flag after the call."""

        _, partitioner = _dummy_setup()
        partitioner._determine_partition_id_to_indices_if_needed()
        self.assertTrue(
            partitioner._partition_id_to_indices_determined
            and len(partitioner._partition_id_to_indices) == 3
        )


class TestSemanticPartitionerFailure(unittest.TestCase):
    """Test SemanticPartitioner failures (exceptions) by incorrect usage."""

    def test_invalid_dataset_type(self):
        """Test if raises when the dataset is not an image dataset."""
        alphabets = list(string.ascii_uppercase)
        data = {
            "letters": [alphabets[i % len(alphabets)] for i in range(300)],
            "label": [i for i in range(300)],
        }
        dataset = Dataset.from_dict(data)
        partitioner = SemanticPartitioner(num_partitions=3, partition_by="label")
        partitioner.dataset = dataset
        with self.assertRaises(TypeError):
            partitioner.load_partition(0)

    @parameterized.expand([((28, 1, 28),), ((3, 3, 32, 32),), ((28,),)])
    def test_invalid_data_shape(self, data_shape: tuple):
        """Test if raises when the data_shape is not a tuple of length 2."""
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(data_shape=data_shape)
            partitioner.load_partition(0)

    @parameterized.expand([(-2,), (-1,), (3,), (4,)])  # type: ignore
    def test_load_invalid_partition_index(self, partition_id):
        """Test if raises when the load_partition is above the num_partitions."""
        _, partitioner = _dummy_setup(num_partitions=3)
        with self.assertRaises(KeyError):
            partitioner.load_partition(partition_id)

    @parameterized.expand([(-1,), (2.5,), (9,), (8), (7.0,)])
    def test_invalid_efficient_net_type(self, efficient_net_type):
        """
        Test if efficient_net_type is not an integer or not in range [0, 7]."""
        with self.assertRaises((ValueError, TypeError)):
            SemanticPartitioner(efficient_net_type=efficient_net_type)

    @parameterized.expand(  # type: ignore
        [(0,), (-1,), (11,), (100,)]
    )  # num_partitions,
    def test_invalid_num_partitions(self, num_partitions):
        """Test if 0 is invalid num_partitions."""
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(num_partitions=num_partitions, num_rows=10)
            partitioner.load_partition(0)

    @parameterized.expand([(0,), (-1,), (2.0,), (11,)])
    def test_invalid_pca_components(self, pca_components):
        """Test if pca_components is not a positive integer."""
        with self.assertRaises((ValueError, TypeError)):
            _, partitioner = _dummy_setup(pca_components=pca_components)
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
                gmm_max_iter=gmm_max_iter, gmm_init_params=gmm_init_params
            )
            partitioner.load_partition(0)


if __name__ == "__main__":
    unittest.main()
