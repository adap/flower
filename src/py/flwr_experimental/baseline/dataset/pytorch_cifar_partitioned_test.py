# Copyright 2020 Adap GmbH. All Rights Reserved.
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
"""Tests for partitioned CIFAR-10 dataset generation."""
# pylint: disable=no-self-use

import unittest
from glob import glob
from os.path import exists, join
from shutil import rmtree
from tempfile import TemporaryDirectory

import numpy as np
import torchvision

from flwr_experimental.baseline.dataset.pytorch_cifar_partitioned import (
    CIFAR10PartitionedDataset,
    augment_dataset,
    generate_partitioned_dataset_files,
)


class CIFAR10PartitionedTestCase(unittest.TestCase):
    """Tests for partitioned CIFAR-10/100 dataset generation."""

    def test_augment_dataset(self) -> None:
        """Test dataset augmentation function."""
        # Load CIFAR10
        tempdir = TemporaryDirectory()
        data_dir = tempdir.name
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=None
        )

        # Define data augmentation transforms
        augment_transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomVerticalFlip(),]
        )

        # Augment existing dataset
        aug_imgs, aug_labels = augment_dataset(
            original_dataset=testset,
            augment_factor=10,
            augment_transform=augment_transform,
        )

        len_dataset = len(testset)
        shape_imgs = testset[0][0].shape
        shape_labels = testset[0][1].shape

        original_data_shapes = (len_dataset, shape_imgs, shape_labels)
        augmented_data_shapes = (*aug_imgs.shape, *aug_labels.shape)

        self.assertSequenceEqual(original_data_shapes, augmented_data_shapes)

    def test_generate_partitioned_dataset_files_nb_partitions(self) -> None:
        """Test if number of partitioned files are being created
        """
        temp_dir = TemporaryDirectory()

        X = np.zeros((5 * 13, 3, 32, 32), dtype=np.uint8)
        Y = np.zeros((5 * 13,), dtype=np.int32)

        fake_dataset = (X, Y)

        generate_partitioned_dataset_files(
            dataset=fake_dataset,
            len_partitions=13,
            nb_partitions=3,
            data_dir=temp_dir.name,
        )

        partitionfiles = [f for f in glob("cifar10*.pt")]
        self.assertEqual(len(partitionfiles), 3)

    def test_generate_partitioned_dataset_files_len_partitions(self) -> None:
        """Test partition function."""
        temp_dir = TemporaryDirectory()

        X = np.zeros((5 * 13, 3, 32, 32), dtype=np.uint8)
        Y = np.zeros((5 * 13,), dtype=np.int32)

        fake_dataset = (X, Y)

        generate_partitioned_dataset_files(
            dataset=fake_dataset,
            len_partitions=13,
            nb_partitions=5,
            data_dir=temp_dir.name,
        )

        XY = torch.load(join(temp_dir.name, "cifar10_0.pt"))

        self.assertSequenceEqual(XY.shape, (13, 5, 3, 32, 32))

    def test_correct_number_of_samples_per_partition(self) -> None:
        """Test if each partition contains the same number of samples for each class"""
        temp_dir = TemporaryDirectory()

        # Load CIFAR10
        trainset = torchvision.datasets.CIFAR10(
            root=temp_dir.name, train=True, download=True, transform=None
        )

        # Define data augmentation transforms
        augment_transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomHorizontalFlip(),]
        )

        # Augment existing dataset and save thumbnails
        augmented_CIFAR10 = augment_dataset(
            original_dataset=trainset,
            augment_factor=10,
            augment_transform=augment_transform,
        )

        # Generate the partionioned files
        generate_partitioned_dataset_files(
            dataset=augmented_CIFAR10,
            len_partitions=500,
            nb_partitions=1000,
            data_dir=temp_dir.name,
        )

        failures = []
        bins = [*range(11)]
        for idx in range(1000):
            X, Y = torch.load(join(temp_dir.name, f"cifar10_{idx}.pt"))
            if (np.bincount(Y) == 50 * np.ones(10)).all():
                failures.append(False)
        self.assertEqual([], failures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
