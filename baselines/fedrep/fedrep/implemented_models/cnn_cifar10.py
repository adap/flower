"""CNNCifar10-v1 model, model manager and model split."""

from typing import Dict, List, Optional, Tuple, Union

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
        client_save_path: Optional[str] = "",
        learning_rate: float = 0.01,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
        """
        super().__init__(
            model_split_class=CNNCifar10ModelSplit, client_id=client_id, config=config
        )
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config["server_device"]
        self.client_save_path = client_save_path if client_save_path != "" else None
        self.learning_rate = learning_rate

    def _create_model(self) -> nn.Module:
        """Return CNNCifar10-v1 model to be splitted into head and body."""
        try:
            return CNNCifar10().to(self.device)
        except:
            self.device = self.config.server_device
            return CNNCifar10().to(self.device)

    def train(
        self, epochs: int = 1
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
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
        if self.client_save_path is not None:
            try:
                self.model.head.load_state_dict(torch.load(self.client_save_path))
            except FileNotFoundError:
                print("No client state found, training from scratch.")
                pass

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learning_rate, momentum=0.5
        )
        correct, total = 0, 0
        loss: torch.Tensor = 0.0
        # self.model.train()
        for i in range(epochs):
            if i < epochs - 1:
                self.model.disable_body()
                self.model.enable_head()
            else:
                self.model.enable_body()
                self.model.disable_head()
            for images, labels in self.trainloader:
                optimizer.zero_grad()
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
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
        if self.client_save_path is not None:
            self.model.head.load_state_dict(torch.load(self.client_save_path))

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        # self.model.eval()
        with torch.no_grad():
            for images, labels in self.testloader:
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        print("Test Accuracy: {:.4f}".format(correct / total))

        if self.client_save_path is not None:
            torch.save(self.model.head.state_dict(), self.client_save_path)

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
