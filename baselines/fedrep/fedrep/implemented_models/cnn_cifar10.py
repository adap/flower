"""CNNCifar10 model, model manager and model split."""

from typing import Optional, Tuple

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
        client_save_path: Optional[str] = None,
        learning_rate: float = 0.01,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            trainloader: The train dataloader.
            testloader: The test dataloader.
            client_save_path: The path where the client's head part parameters saved.
            learning_rate: The learning rate.
        """
        super().__init__(
            client_id,
            config,
            trainloader,
            testloader,
            client_save_path,
            learning_rate,
            model_split_class=CNNCifar10ModelSplit,
        )

    def _create_model(self) -> CNNCifar10:
        """Return CNNCifar10 model to be splitted into head and body."""
        try:
            return CNNCifar10().to(self.device)
        except AttributeError:
            self.device = self.config.server_device
            return CNNCifar10().to(self.device)
