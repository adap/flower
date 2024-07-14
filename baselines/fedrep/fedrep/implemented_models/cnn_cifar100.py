"""CNNCifar100 model, model manager and model split."""

from typing import Optional, Tuple

import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedrep.models import ModelManager, ModelSplit


class CNNCifar100(nn.Module):
    """CNN model for CIFAR100 dataset."""

    def __init__(self):
        """Initialize the model."""
        super(CNNCifar100, self).__init__()

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


class CNNCifar100ModelSplit(ModelSplit):
    """Split CNNCifar100 model into body and head."""

    def _get_model_parts(self, model: CNNCifar100) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


class CNNCifar100ModelManager(ModelManager):
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
            model_split_class=CNNCifar100ModelSplit,
        )

    def _create_model(self) -> CNNCifar100:
        """Return CNNCifar100 model to be splitted into head and body."""
        try:
            return CNNCifar100().to(self.device)
        except AttributeError:
            self.device = self.config["server_device"]
            return CNNCifar100().to(self.device)
