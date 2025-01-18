"""fedrep: A Flower Baseline."""

from typing import Tuple

import torch
from torch import nn

from .base_model import ModelManager, ModelSplit


# pylint: disable=W0223
class CNNCifar10(nn.Module):
    """CNN model for CIFAR10 dataset.

    Refer to
    https://github.com/rahulv0205/fedrep_experiments/blob/main/models/Nets.py
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__()

        # Note that in the official implementation, the body has no BN layers.
        # However, no BN will definitely lead training to collapse.
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 64),
            nn.ReLU(),
        )

        self.head = nn.Sequential(nn.Linear(64, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.body(x)
        return self.head(x)


class CNNCifar10ModelSplit(ModelSplit):
    """Split CNNCifar10 model into body and head."""

    def _get_model_parts(self, model: CNNCifar10) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


# pylint: disable=R0902, R0913, R0801
class CNNCifar10ModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(self, **kwargs):
        """Initialize the attributes of the model manager."""
        super().__init__(model_split_class=CNNCifar10ModelSplit, **kwargs)

    def _create_model(self) -> nn.Module:
        """Return CNNCifar10 model to be split into head and body."""
        return CNNCifar10()


# pylint: disable=W0223
class CNNCifar100(nn.Module):
    """CNN model for CIFAR100 dataset.

    Refer to
    https://github.com/rahulv0205/fedrep_experiments/blob/main/models/Nets.py
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__()

        # Note that in the official implementation, the body has no BN layers.
        # However, no BN will definitely lead training to collapse.
        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(128 * 5 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.6),
        )

        self.head = nn.Sequential(nn.Linear(128, 100))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.body(x)
        return self.head(x)


class CNNCifar100ModelSplit(ModelSplit):
    """Split CNNCifar100 model into body and head."""

    def _get_model_parts(self, model: CNNCifar100) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


# pylint: disable=R0902, R0913, R0801
class CNNCifar100ModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(self, **kwargs):
        """Initialize the attributes of the model manager."""
        super().__init__(model_split_class=CNNCifar100ModelSplit, **kwargs)

    def _create_model(self) -> CNNCifar100:
        """Return CNNCifar100 model to be split into head and body."""
        return CNNCifar100()
