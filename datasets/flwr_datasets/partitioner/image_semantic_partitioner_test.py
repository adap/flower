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
"""Test ImageSemanticPartitioner."""


# pylint: disable=W0212
import string
import unittest
from typing import Optional, Tuple

import numpy as np
from parameterized import parameterized

from datasets import Dataset
from flwr_datasets.partitioner.image_semantic_partitioner import (
    ImageSemanticPartitioner,
)


# pylint: disable=R0913
def _dummy_setup(
    data_shape: Tuple[int, ...] = (28, 28, 1),
    num_partitions: int = 3,
    num_rows: int = 10,
    partition_by: str = "label",
    efficient_net_type: int = 0,
    batch_size: int = 32,
    pca_components: int = 6,
    gmm_max_iter: int = 2,
    gmm_init_params: str = "random",
    sample_seed: int = 42,
    pca_seed: Optional[int] = None,
    gmm_seed: Optional[int] = None,
) -> Tuple[Dataset, ImageSemanticPartitioner]:
    """Create a dummy dataset and partitioner for testing."""
    data = {
        "image": [np.random.randn(*data_shape) for _ in range(num_rows)],
        "label": [i % 3 for i in range(num_rows)],
    }
    dataset = Dataset.from_dict(data)
    partitioner = ImageSemanticPartitioner(
        num_partitions=num_partitions,
        partition_by=partition_by,
        efficient_net_type=efficient_net_type,
        batch_size=batch_size,
        pca_components=pca_components,
        gmm_max_iter=gmm_max_iter,
        gmm_init_params=gmm_init_params,
        sample_seed=sample_seed,
        pca_seed=pca_seed,
        gmm_seed=gmm_seed,
    )
    partitioner.dataset = dataset
    return dataset, partitioner


class TestImageSemanticPartitionerSuccess(unittest.TestCase):
    """Test ImageSemanticPartitioner used with no exceptions."""

    # pylint: disable=R0913, R0914
    @parameterized.expand(  # type: ignore
        [
            ((28, 28, 1), 3, 50, "label", 0, 32, 128, 2, "kmeans", 1, 2, 3),
            ((1, 28, 28), 5, 100, "label", 2, 64, 256, 1, "random", 1, 2, 3),
            ((32, 32, 3), 5, 100, "label", 7, 16, 256, 1, "k-means++", 1, 2, 3),
        ]
    )
    def test_valid_initialization(
        self,
        data_shape: Tuple[int],
        num_partitions: int,
        num_rows: int,
        partition_by: str,
        efficient_net_type: int,
        batch_size: int,
        pca_components: int,
        gmm_max_iter: int,
        gmm_init_params: str,
        sample_seed: int,
        pca_seed: Optional[int] = None,
        gmm_seed: Optional[int] = None,
    ) -> None:
        """Test whether initializaiton is successful."""
        _, partitioner = _dummy_setup(
            data_shape=data_shape,
            num_partitions=num_partitions,
            num_rows=num_rows,
            partition_by=partition_by,
            efficient_net_type=efficient_net_type,
            batch_size=batch_size,
            pca_components=pca_components,
            gmm_max_iter=gmm_max_iter,
            gmm_init_params=gmm_init_params,
            sample_seed=sample_seed,
            pca_seed=pca_seed,
            gmm_seed=gmm_seed,
        )
        _sample_seed = sample_seed
        _pca_seed = pca_seed if pca_seed is not None else sample_seed
        _gmm_seed = gmm_seed if gmm_seed is not None else sample_seed
        self.assertEqual(
            (
                partitioner._num_partitions,
                partitioner._partition_by,
                partitioner._efficient_net_type,
                partitioner._pca_components,
                partitioner._gmm_max_iter,
                partitioner._gmm_init_params,
                partitioner._sample_seed,
                partitioner._pca_seed,
                partitioner._gmm_seed,
            ),
            (
                num_partitions,
                partition_by,
                efficient_net_type,
                pca_components,
                gmm_max_iter,
                gmm_init_params,
                _sample_seed,
                _pca_seed,
                _gmm_seed,
            ),
        )

    @parameterized.expand([((28, 28, 1),), ((3, 32, 32),), ((28, 28),)])  # type: ignore
    def test_data_shape(self, data_shape: Tuple[int]) -> None:
        """Test if data_shape is correct."""
        _, partitioner = _dummy_setup(data_shape=data_shape)
        partitioner.load_partition(0)

    @parameterized.expand([(0,), (3,), (7,)])  # type: ignore
    def test_efficient_net(self, efficient_net_type: int) -> None:
        """Test if efficient_net_type is correct."""
        _, partitioner = _dummy_setup(efficient_net_type=efficient_net_type)
        partitioner.load_partition(0)

    @parameterized.expand([(64,), (96,), (32,)])  # type: ignore
    def test_pca_components(self, pca_components: int) -> None:
        """Test if pca_components is correct."""
        _, partitioner = _dummy_setup(num_rows=100, pca_components=pca_components)
        self.assertEqual(partitioner._pca_components, pca_components)

    @parameterized.expand(  # type: ignore
        [(1, "random"), (2, "kmeans"), (2, "k-means++")]
    )
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

    @parameterized.expand([(1, None, None), (1, 2, None), (1, 2, 3)])  # type: ignore
    def test_seeds(
        self, sample_seed: int, pca_seed: Optional[int], gmm_seed: Optional[int]
    ) -> None:
        """Test if seeds are correct."""
        _sample_seed = sample_seed
        _pca_seed = pca_seed if pca_seed is not None else sample_seed
        _gmm_seed = gmm_seed if gmm_seed is not None else sample_seed
        _, partitioner = _dummy_setup(
            sample_seed=sample_seed, pca_seed=pca_seed, gmm_seed=gmm_seed
        )
        self.assertEqual(
            (partitioner._sample_seed, partitioner._pca_seed, partitioner._gmm_seed),
            (_sample_seed, _pca_seed, _gmm_seed),
        )

    def test_determine_partition_id_to_indices(self) -> None:
        """Test the determine_nod_id_to_indices matches the flag after the call."""
        _, partitioner = _dummy_setup()
        partitioner._determine_partition_id_to_indices_if_needed()
        self.assertTrue(
            partitioner._partition_id_to_indices_determined
            and len(partitioner._partition_id_to_indices) == 3
        )


class TestImageSemanticPartitionerFailure(unittest.TestCase):
    """Test ImageSemanticPartitioner failures (exceptions) by incorrect usage."""

    def test_invalid_dataset_type(self) -> None:
        """Test if raises when the dataset is not an image dataset."""
        alphabets = list(string.ascii_uppercase)
        data = {
            "letters": [alphabets[i % len(alphabets)] for i in range(300)],
            "label": list(range(300)),
        }
        dataset = Dataset.from_dict(data)
        partitioner = ImageSemanticPartitioner(num_partitions=3, partition_by="label")
        partitioner.dataset = dataset
        with self.assertRaises((TypeError, ValueError)):
            partitioner.load_partition(0)

    @parameterized.expand([(0,), (-1,)])  # type: ignore
    def test_invalid_batch_size(self, batch_size: int) -> None:
        """Test if raises when the batch_size is not a positive integer."""
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(batch_size=batch_size)
            partitioner.load_partition(0)

    @parameterized.expand([((28, 1, 28),), ((3, 3, 32, 32),), ((28,),)])  # type: ignore
    def test_invalid_data_shape(self, data_shape: Tuple[int]) -> None:
        """Test if raises when the data_shape is not a tuple of length 2."""
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(data_shape=data_shape)
            partitioner.load_partition(0)

    @parameterized.expand([(-2,), (-1,), (3,), (4,)])  # type: ignore
    def test_load_invalid_partition_index(self, partition_id: int) -> None:
        """Test if raises when the load_partition is above the num_partitions."""
        _, partitioner = _dummy_setup(num_partitions=3)
        with self.assertRaises(KeyError):
            partitioner.load_partition(partition_id)

    @parameterized.expand([(-1,), (2.5,), (9,), (8), (7.0,)])  # type: ignore
    def test_invalid_efficient_net_type(self, efficient_net_type: int) -> None:
        """Test if efficient_net_type is not an integer or not in range [0, 7]."""
        with self.assertRaises((ValueError, TypeError)):
            _dummy_setup(efficient_net_type=efficient_net_type)

    @parameterized.expand([(0,), (-1,), (11,), (100,)])  # type: ignore
    def test_invalid_num_partitions(self, num_partitions: int) -> None:
        """Test if 0 is invalid num_partitions."""
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(num_partitions=num_partitions, num_rows=10)
            partitioner.load_partition(0)

    @parameterized.expand([(0,), (-1,), (2.0,), (11,)])  # type: ignore
    def test_invalid_pca_components(self, pca_components: int) -> None:
        """Test if pca_components is not a positive integer."""
        with self.assertRaises((ValueError, TypeError)):
            _, partitioner = _dummy_setup(pca_components=pca_components)
            partitioner.load_partition(0)

    @parameterized.expand(  # type: ignore
        [
            (0, "random"),
            (-1, "keams"),
            (2.0, "k-means++"),
            (2, "rand"),
            (3, "kmean"),
            (10, "kmeans++"),
        ]
    )
    def test_invalid_gaussian_mixture_config(
        self, gmm_max_iter: int, gmm_init_params: str
    ) -> None:
        """Test if gmm_max_iter and gmm_init_params are not valid."""
        with self.assertRaises(ValueError):
            _, partitioner = _dummy_setup(
                gmm_max_iter=gmm_max_iter, gmm_init_params=gmm_init_params
            )
            partitioner.load_partition(0)

    @parameterized.expand(  # type: ignore
        [("1", None, None), (1.2, 2, None), (None, 2, 3)]
    )
    def test_invalid_seeds(
        self, sample_seed: int, pca_seed: Optional[int], gmm_seed: Optional[int]
    ) -> None:
        """Test if raises when the seeds are not integers."""
        with self.assertRaises((TypeError, ValueError)):
            _, partitioner = _dummy_setup(
                sample_seed=sample_seed, pca_seed=pca_seed, gmm_seed=gmm_seed
            )
            partitioner.load_partition(0)


if __name__ == "__main__":
    unittest.main()
