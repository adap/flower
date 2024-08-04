"""CNNCifar10-v1 model, model manager and model split."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedrep.models import ModelManager, ModelSplit


class CNNCifar10(nn.Module):
    """CNN model for CIFAR10 dataset.

    Refer to
    https://github.com/rahulv0205/fedrep_experiments/blob/main/models/Nets.py
    """

    def __init__(self):
        """Initialize the model."""
        super(CNNCifar10, self).__init__()

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


class CNNCifar10ModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        client_save_path: Optional[str],
        learning_rate: float = 0.01,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
        """
        super().__init__(
            model_split_class=CNNCifar10ModelSplit,
            client_id=client_id,
            config=config,
            trainloader=trainloader,
            testloader=testloader,
            client_save_path=client_save_path,
            learning_rate=learning_rate,
        )

    def _create_model(self) -> nn.Module:
        """Return CNNCifar10-v1 model to be splitted into head and body."""
        return CNNCifar10().to(self.device)
