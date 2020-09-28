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
# pylint: disable=invalid-name

import random
from os import path
from pathlib import Path
from typing import List
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as TF
from torch import from_numpy
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from flwr_experimental.baseline.dataset.dataset import (
    XY,
    shuffle,
    sort_by_label,
    sort_by_label_repeating,
)


class Rot90Transform:
    """Rotates image by a multiple of 90 degrees"""

    def __init__(self, angles: List[int]):
        self.angles = angles

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


def augment_dataset(
    original_dataset: torch.utils.data.Dataset,
    augment_factor: int,
    augment_transform: torchvision.transforms,
    save_thumbnails_dir: Optional[str] = None,
) -> XY:
    """Augments the dataset augment_transform

    Dataset is augmented to len(original_dataset)*augment_factor
    using augment_transform. The original dataset will be a subset
    of the new one.

    Parameters
    ----------
    original_dataset : torch.utils.data.Dataset
        Original PyTorch Dataset to be augmented
    augment_factor : int
        Number of augmented copies generated from each original sample.
        This will multiply the final length of the dataset by augment_factor.
    augment_transform : torchvision.transforms
        Composition of transforms used to augment the images.
        Last transform in this composition does not need to be a ToTensor.
    save_thumbnails_dir
        Directory where to save thumbnails. One jpg image will be created for each original image
        with a augment_factor thumbnails generated.

    Returns
    -------
    XY
        A tuple containing the new augmented dataset
    """
    # Create directory if it does not exist
    if save_thumbnails_dir:
        save_thumbnails_dir = path.expanduser(save_thumbnails_dir)
        Path(save_thumbnails_dir).mkdir(parents=True, exist_ok=True)

    combined_transform = torchvision.transforms.Compose(
        [augment_transform, torchvision.transforms.ToTensor()]
    )

    augmented_imgs = np.empty(
        (len(original_dataset), augment_factor, 3, 32, 32), dtype=np.float32
    )
    augmented_labels = np.empty((len(original_dataset), augment_factor), dtype=np.long)

    print("Generating augmented images...")
    for idx, (img, label) in enumerate(tqdm(original_dataset)):
        augmented_imgs[idx, 0, :, :, :] = torchvision.transforms.ToTensor()(img)
        augmented_labels[idx, :] = label

        for j in range(1, augment_factor):
            augmented_imgs[idx, j, :, :, :] = combined_transform(img)

        # Save augmented set disk as thumbnails?
        if save_thumbnails_dir is not None:
            if path.isdir(save_thumbnails_dir):
                tmp = [*augmented_imgs[idx, :, :, :, :]]
                tmp2 = np.concatenate(tmp, axis=-1)
                save_image(
                    from_numpy(tmp2), path.join(save_thumbnails_dir, f"cifar10_{idx}.jpg")
                )

    # Rearrange dataset. Every multiple of augment_factor is an original image.
    X = augmented_imgs.reshape((-1, 3, 32, 32))
    Y = augmented_labels.reshape(-1)
    return (X, Y)


def generate_partitioned_dataset_files(
    dataset: XY,
    len_partitions: int,
    nb_partitions: int,
    data_dir: str,
) -> None:
    """Generates a set of tensor files contaning partitions of dataset.

    Parameters
    ----------
    dataset:
        Dataset in the form XY to be partitioned.
    len_partitions : int
       Number of samples inside a partition.
    nb_partitions : int
        Total number of partitions to be generated.
    data_dir: str
        Path to directory where the partition dataset files will be stored.
    """

    X, Y = dataset

    X, Y = shuffle(X, Y)
    X, Y = sort_by_label_repeating(X, Y)

    # Create one file per partition
    for part_idx in range(nb_partitions):
        run_idx = [*range(part_idx * len_partitions, (part_idx + 1) * len_partitions)]
        this_part_X = X[run_idx]
        this_part_Y = Y[run_idx]
        torch.save((this_part_X, this_part_Y), path.join(data_dir, f"cifar10_{part_idx}.pt"))


class CIFAR10PartitionedDataset(torch.utils.data.Dataset):
    """Augmented and partitioned dataset based on CIFAR10."""

    def __init__(
        self, partition_id: int, root_dir: str, transform: torchvision.transforms = None
    ):
        """Dataset from partitioned files

        Parameters
        ----------
        partition_id : int
            Partition file ID. Usually the same as the client ID.
        root_dir : str
            Directory containing partioned files.
        transform : torchvision.transform
            Transforms to be applied (usually normalization) to tensors before creating dataset.
        """
        self.partition_id = partition_id
        self.root_dir = root_dir
        self.partition_path = path.join(self.root_dir, f"cifar10_{self.partition_id}.pt")
        self.transform = (
            transforms.Compose([transforms.ToPILImage(), transform])
            if transform
            else None
        )

        if not path.exists(self.partition_path):
            raise RuntimeError(f"Partition file {self.partition_path} not found.")
        else:
            self.X, self.Y = torch.load(self.partition_path)
            self.X = torch.from_numpy(self.X)
            self.Y = torch.from_numpy(self.Y)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> XY:
        x = self.X[idx]
        y = self.Y[idx]
        if self.transform:
            x = self.transform(x)

        return (x, y)


if __name__ == "__main__":
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
    save_thumbnails_dir = path.join(data_dir, "thumbnails")
    augmented_CIFAR10 = augment_dataset(
        original_dataset=trainset,
        augment_factor=10,
        augment_transform=augment_transform,
        save_thumbnails_dir=save_thumbnails_dir,
    )

    # Generate the partionioned files
    generate_partitioned_dataset_files(
        dataset=augmented_CIFAR10,
        len_partitions=500,
        nb_partitions=1000,
        data_dir=data_dir,
    )

    # Generate the dataset from saved files
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    specific_augmented_dataset = CIFAR10PartitionedDataset(
        partition_id=0, root_dir=data_dir, transform=transform
    )
