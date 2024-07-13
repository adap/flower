"""CNNCifar100 model, model manager and model split."""

from typing import Dict, List, Optional, Tuple, Union

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

        self.body = nn.Sequential(
            nn.Conv2d(3, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 5),
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

    def forward(self, x):
        """Forward pass of the model."""
        x = self.body(x)
        x = self.head(x)
        # Basically we don't need to do log softmax explicitly,
        # as it is done by nn.CrossEntropyLoss()
        # So we actually should just return the x directly.
        # return x

        # However the official implementation did that,
        # so I leave this as it is.
        return torch.nn.functional.log_softmax(x, dim=1)


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
        client_save_path: Optional[str] = "",
        learning_rate: float = 0.1,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
        """
        super().__init__(
            model_split_class=CNNCifar100ModelSplit, client_id=client_id, config=config
        )
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config.server_device
        self.client_save_path = client_save_path if client_save_path != "" else None
        self.learning_rate = learning_rate

    def _create_model(self) -> CNNCifar100:
        """Return CNNCifar100 model to be splitted into head and body."""
        try:
            return CNNCifar100().to(self.device)
        except AttributeError:
            self.device = self.config["server_device"]
            return CNNCifar100().to(self.device)

    def train(
        self, epochs: int = 5, rep_epochs: int = 1
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Method adapted from
        https://github.com/rahulv0205/fedrep_experiments/blob/main/models/Nets.py.

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
        head_epochs = epochs - rep_epochs
        criterion = torch.nn.CrossEntropyLoss()
        weights = [v for k, v in self.model.named_parameters() if "weight" in k]
        biases = [v for k, v in self.model.named_parameters() if "bias" in k]
        optimizer = torch.optim.SGD(
            [
                {"params": weights, "weight_decay": 0.0001},
                {"params": biases, "weight_decay": 0},
            ],
            lr=self.learning_rate,
            momentum=0.5,
        )
        correct, total = 0, 0
        loss: torch.Tensor = 0.0
        self.model.train()
        for i in range(epochs):
            if self.config.algorithm == "fedrep":
                if i < head_epochs:
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
                correct += (torch.argmax(outputs.data, 1) == labels).sum().item()

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
        self.model.eval()
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
        return len(self.trainloader)

    def test_dataset_size(self) -> int:
        """Return test data set size."""
        return len(self.testloader)

    def total_dataset_size(self) -> int:
        """Return total data set size."""
        return len(self.trainloader) + len(self.testloader)
