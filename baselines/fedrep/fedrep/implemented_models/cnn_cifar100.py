"""CNNCifar100 model, model manager and model split."""

import os
from typing import Dict, List, Optional, Tuple, Union

from fedrep.constants import (
    DEFAULT_LOCAL_TRAIN_EPOCHS,
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_REPRESENTATION_EPOCHS,
)
import torch
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

    def train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Method adapted from simple CNNCifar10-v1 (PyTorch) \
        https://github.com/wjc852456/pytorch-cnncifar10net-v1.

        Args:
            epochs: number of training epochs.

        Returns
        -------
            Dict containing the train metrics.
        """
        # Load client state (head) if client_save_path is not None and it is not empty
        if self.client_save_path is not None and os.path.isfile(self.client_save_path):
            self.model.head.load_state_dict(torch.load(self.client_save_path))

        num_local_epochs = DEFAULT_LOCAL_TRAIN_EPOCHS
        if hasattr(self.config, "num_local_epochs"):
            num_local_epochs = int(self.config.num_local_epochs)

        num_rep_epochs = DEFAULT_REPRESENTATION_EPOCHS
        if hasattr(self.config, "num_rep_epochs"):
            num_rep_epochs = int(self.config.num_rep_epochs)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.5
        )
        correct, total = 0, 0
        loss: torch.Tensor = 0.0

        self.model.train()
        for i in range(num_local_epochs + num_rep_epochs):
            if i < num_local_epochs:
                self.model.disable_body()
                self.model.enable_head()
            else:
                self.model.enable_body()
                self.model.disable_head()
            for images, labels in self.trainloader:
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        # Save client state (head)
        if self.client_save_path is not None:
            torch.save(self.model.head.state_dict(), self.client_save_path)

        return {"loss": loss.item(), "accuracy": correct / total}

    def test(self) -> Dict[str, float]:
        """Test the model maintained in self.model.

        Returns
        -------
            Dict containing the test metrics.
        """
        # Load client state (head)
        if self.client_save_path is not None and os.path.isfile(self.client_save_path):
            self.model.head.load_state_dict(torch.load(self.client_save_path))

        num_finetune_epochs = DEFAULT_FINETUNE_EPOCHS
        if hasattr(self.config, "num_finetune_epochs"):
            num_finetune_epochs = int(self.config.num_finetune_epochs)

        if num_finetune_epochs > 0:
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            criterion = torch.nn.CrossEntropyLoss()
            self.model.train()
            for _ in range(num_finetune_epochs):
                for images, labels in self.trainloader:
                    outputs = self.model(images.to(self.device))
                    labels = labels.to(self.device)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0

        self.model.eval()
        with torch.no_grad():
            for images, labels in self.testloader:
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        return {
            "loss": loss / len(self.testloader.dataset),
            "accuracy": correct / total,
        }

    def train_dataset_size(self) -> int:
        """Return train data set size."""
        return len(self.trainloader.dataset)

    def test_dataset_size(self) -> int:
        """Return test data set size."""
        return len(self.testloader.dataset)

    def total_dataset_size(self) -> int:
        """Return total data set size."""
        return len(self.trainloader.dataset) + len(self.testloader.dataset)

    def _create_model(self) -> CNNCifar100:
        """Return CNNCifar100 model to be splitted into head and body."""
        return CNNCifar100().to(self.device)
