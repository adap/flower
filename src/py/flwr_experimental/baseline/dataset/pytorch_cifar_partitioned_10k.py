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
"""Partitioned versions of CIFAR-10 datasets."""
from os import path

import torchvision
from torchvision import transforms

from .pytorch_cifar_partitioned import (
    Rot90Transform,
    generate_partitioned_dataset_files,
    augment_dataset,
    CIFAR10PartitionedDataset,
)


def main():
    # Where to save partitions
    data_dir = path.expanduser("~/.flower/data/cifar-10")
    # Load CIFAR10
    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=None
    )

    # Define data augmentation transforms
    augment_transform = torchvision.transforms.Compose(
        [
            Rot90Transform(angles=[-30, -15, 0, 15, 30]),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ColorJitter(
                brightness=0.05, contrast=0.3, saturation=0.6, hue=0.08
            ),
        ]
    )

    # Augment existing dataset and save thumbnails
    augmented_CIFAR10 = augment_dataset(
        original_dataset=trainset,
        augment_factor=100,
        augment_transform=augment_transform,
    )

    # Generate the partitioned files
    generate_partitioned_dataset_files(
        dataset=augmented_CIFAR10,
        len_partitions=500,
        nb_partitions=10000,
        data_dir=data_dir,
    )

    # Generate the dataset from saved files
    # transform = transforms.Compose(
    #     [
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     ]
    # )
    # specific_augmented_dataset = CIFAR10PartitionedDataset(
    #     partition_id=0, root_dir=data_dir, transform=transform
    # )


if __name__ == "__main__":
    main()
