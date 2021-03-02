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
"""PyTorch CIFAR-10/100 image classification."""

# mypy: ignore-errors
# pylint: disable=W0223

from os import PathLike
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image
from PIL.Image import Image as ImageType
from torch import load
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from flwr.dataset.utils.common import XY


def get_normalization_transform(num_classes: int = 10) -> Compose:
    """Generates a compose transformation with mean and average normalization
    for CIFAR10.

    Returns:
        transforms.transforms.Compose: A Compose transformation for CIFAR10
    """
    if num_classes not in [10, 100]:
        raise ValueError(
            """Number of classes can only be either
                10 or 100 for CIFAR10 and CIFAR100 datasets respectively."""
        )
    if num_classes == 10:
        mean_std = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        mean_std = ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform = Compose(
        [
            ToTensor(),
            Normalize(*mean_std),
        ]
    )
    return transform


class CIFARPartitionedDataset(Dataset):
    """Defines PyTorch Dataset that reads from a Flower dataset partition.

    Args:
        Dataset (torch.utils.data.Dataset): PyTorch dataset.
    """

    def __init__(
        self,
        *,
        num_classes: int = 10,
        root_dir: Union[str, bytes, PathLike],
        partition_id: int,
        transform: Optional[callable] = None,
    ):
        """Dataset from partitioned files
        Parameters
        ----------
        num_classes: int
            Defines which dataset to use. CIFAR10 or CIFAR100.
        partition_id : int
            Partition file ID. Usually the same as the client ID.
        root_dir : Union[str, bytes, os.PathLike]
            Directory containing partioned files.
        """

        if num_classes not in [10, 100]:
            raise ValueError(
                """Number of classes can only be either
                10 or 100 for CIFAR10 and CIFAR100 datasets respectively."""
            )
        self.root_dir: Path = Path(root_dir).expanduser()
        self.partition_id: int = partition_id
        self.partition_path = self.root_dir / f"{self.partition_id}.pt"

        if not self.partition_path.exists():
            raise RuntimeError(f"Partition file {self.partition_path} not found.")

        self.images, self.labels = load(self.partition_path)

        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Union[XY, Tuple[ImageType, np.ndarray]]:
        image = Image.fromarray(self.images[idx])
        label = self.labels.item(idx)

        if self.transform:
            image = self.transform(image)

        return (image, label)


class CIFAR10PartitionedDataset(CIFARPartitionedDataset):
    """Augmented and partitioned dataset based on CIFAR10."""

    def __init__(
        self,
        *,
        root_dir: Union[str, bytes, PathLike],
        partition_id: int,
        transform: Optional[callable] = None,
    ):
        """Dataset from partitioned files
        Parameters
        ----------
        partition_id : int
            Partition file ID. Usually the same as the client ID.
        root_dir : Union[str, bytes, os.PathLike]
            Directory containing partioned files.
        """
        super().__init__(
            num_classes=10,
            root_dir=root_dir,
            partition_id=partition_id,
            transform=transform,
        )


class CIFAR100PartitionedDataset(CIFARPartitionedDataset):
    """Augmented and partitioned dataset based on CIFAR10."""

    def __init__(
        self,
        *,
        root_dir: Union[str, bytes, PathLike],
        partition_id: int,
        transform: Optional[callable] = None,
    ):
        """Dataset from partitioned files
        Parameters
        ----------
        partition_id : int
            Partition file ID. Usually the same as the client ID.
        root_dir : Union[str, bytes, os.PathLike]
            Directory containing partioned files.
        """
        super().__init__(
            num_classes=100,
            root_dir=root_dir,
            partition_id=partition_id,
            transform=transform,
        )
