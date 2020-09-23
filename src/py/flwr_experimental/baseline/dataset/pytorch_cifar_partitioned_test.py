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
from tqdm import tqdm

import numpy as np
import torch
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
        with TemporaryDirectory() as temp_dir:
            testset = torchvision.datasets.CIFAR10(
                root=temp_dir, train=False, download=True, transform=None
            )

            len_dataset = len(testset)
            shape_imgs = testset[0][0].size
            shape_labels = 1 #It's an int

            # Define data augmentation transforms
            augment_transform = torchvision.transforms.Compose(
                [torchvision.transforms.RandomVerticalFlip(),
                torchvision.transforms.RandomHorizontalFlip()]
            )

            # Augment existing dataset
            augment_factor = 10
            aug_imgs, aug_labels = augment_dataset(
                original_dataset=testset,
                augment_factor=augment_factor,
                augment_transform=augment_transform,
            )

            self.assertEqual(len(aug_imgs), augment_factor*len(testset))
            self.assertEqual(len(aug_labels), augment_factor*len(testset))
            self.assertSequenceEqual(testset[0][0].size, aug_imgs[0].shape[1:])

    def test_generate_partitioned_dataset(self) -> None:
        """Tests if partitions are being created properly"""
        with TemporaryDirectory() as temp_dir:
            X = np.zeros((5 * 13, 3, 32, 32), dtype=np.uint8)
            Y = np.zeros((5 * 13,), dtype=np.int32)

            fake_dataset = (X, Y)
            len_partitions=13
            nb_partitions = 5
            generate_partitioned_dataset_files(
                dataset=fake_dataset,
                len_partitions=len_partitions,
                nb_partitions=nb_partitions,
                data_dir=temp_dir
            )

            partitionfiles = [f for f in glob(join(temp_dir, "cifar10*.pt"))]
            self.assertEqual(len(partitionfiles), nb_partitions, "Number of files must be equal to the number of generated partitions")

            X, Y = torch.load(join(temp_dir, "cifar10_0.pt"))
            self.assertEqual(Y.shape[0], (len_partitions), "length of partition must be equal to len_partition")
            self.assertSequenceEqual(X.shape, (13, 3, 32, 32))

    def test_uniform_distribution_within_partition(self) -> None:
        """Test if each partition contains the same number of samples for each class"""
        with TemporaryDirectory() as temp_dir:
            # Load CIFAR10
            trainset = torchvision.datasets.CIFAR10(
                root=temp_dir, train=True, download=True, transform=None
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
                data_dir=temp_dir,
            )

            for idx in tqdm(range(1000)):
                X, Y = torch.load(join(temp_dir, f"cifar10_{idx}.pt"))
                is_label_hist_uniform = (np.bincount(Y) == 50 * np.ones(10)).all()

                self.assertTrue(is_label_hist_uniform)


if __name__ == "__main__":
    unittest.main(verbosity=2)
