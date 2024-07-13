"""MobileNet-v1 model, model manager and model split."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedper.models import ModelManager, ModelSplit

# Set model architecture
ARCHITECTURE = {
    "layer_1": {"conv_dw": [32, 64, 1]},
    "layer_2": {"conv_dw": [64, 128, 2]},
    "layer_3": {"conv_dw": [128, 128, 1]},
    "layer_4": {"conv_dw": [128, 256, 2]},
    "layer_5": {"conv_dw": [256, 256, 1]},
    "layer_6": {"conv_dw": [256, 512, 2]},
    "layer_7": {"conv_dw": [512, 512, 1]},
    "layer_8": {"conv_dw": [512, 512, 1]},
    "layer_9": {"conv_dw": [512, 512, 1]},
    "layer_10": {"conv_dw": [512, 512, 1]},
    "layer_11": {"conv_dw": [512, 512, 1]},
    "layer_12": {"conv_dw": [512, 1024, 2]},
    "layer_13": {"conv_dw": [1024, 1024, 1]},
}


class MobileNet(nn.Module):
    """Model from MobileNet-v1 (https://github.com/wjc852456/pytorch-mobilenet-v1)."""

    def __init__(
        self,
        num_head_layers: int = 1,
        num_classes: int = 10,
    ) -> None:
        super(MobileNet, self).__init__()

        self.architecture = ARCHITECTURE

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.body = nn.Sequential()
        self.body.add_module("initial_batch_norm", conv_bn(3, 32, 2))
        for i in range(1, 13):
            for _, value in self.architecture[f"layer_{i}"].items():
                self.body.add_module(f"conv_dw_{i}", conv_dw(*value))

        self.body.add_module("avg_pool", nn.AvgPool2d([7]))
        self.body.add_module("fc", nn.Linear(1024, num_classes))

        if num_head_layers == 1:
            self.head = nn.Sequential(
                nn.AvgPool2d([7]), nn.Flatten(), nn.Linear(1024, num_classes)
            )
            self.body.avg_pool = nn.Identity()
            self.body.fc = nn.Identity()
        elif num_head_layers == 2:
            self.head = nn.Sequential(
                conv_dw(1024, 1024, 1),
                nn.AvgPool2d([7]),
                nn.Flatten(),
                nn.Linear(1024, num_classes),
            )
            self.body.conv_dw_13 = nn.Identity()
            self.body.avg_pool = nn.Identity()
            self.body.fc = nn.Identity()
        elif num_head_layers == 3:
            self.head = nn.Sequential(
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
                nn.AvgPool2d([7]),
                nn.Flatten(),
                nn.Linear(1024, num_classes),
            )
            self.body.conv_dw_12 = nn.Identity()
            self.body.conv_dw_13 = nn.Identity()
            self.body.avg_pool = nn.Identity()
            self.body.fc = nn.Identity()
        elif num_head_layers == 4:
            self.head = nn.Sequential(
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
                nn.AvgPool2d([7]),
                nn.Flatten(),
                nn.Linear(1024, num_classes),
            )
            self.body.conv_dw_11 = nn.Identity()
            self.body.conv_dw_12 = nn.Identity()
            self.body.conv_dw_13 = nn.Identity()
            self.body.avg_pool = nn.Identity()
            self.body.fc = nn.Identity()
        else:
            raise NotImplementedError("Number of head layers not implemented.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        x = self.body(x)
        return self.head(x)


class MobileNetModelSplit(ModelSplit):
    """Split MobileNet model into body and head."""

    def _get_model_parts(self, model: MobileNet) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


class MobileNetModelManager(ModelManager):
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
            model_split_class=MobileNetModelSplit,
            client_id=client_id,
            config=config,
        )
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config["server_device"]
        self.client_save_path = client_save_path if client_save_path != "" else None
        self.learning_rate = learning_rate

    def _create_model(self) -> nn.Module:
        """Return MobileNet-v1 model to be splitted into head and body."""
        try:
            return MobileNet(
                num_head_layers=self.config["model"]["num_head_layers"],
                num_classes=self.config["model"]["num_classes"],
            ).to(self.device)
        except AttributeError:
            self.device = self.config["server_device"]
            return MobileNet(
                num_head_layers=self.config["model"]["num_head_layers"],
                num_classes=self.config["model"]["num_classes"],
            ).to(self.device)

    def train(
        self,
        epochs: int = 1,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Method adapted from simple MobileNet-v1 (PyTorch) \
        https://github.com/wjc852456/pytorch-mobilenet-v1.

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
            self.model.parameters(), lr=self.learning_rate, momentum=0.9
        )
        correct, total = 0, 0
        loss: torch.Tensor = 0.0
        # self.model.train()
        for _ in range(epochs):
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

    def test(
        self,
    ) -> Dict[str, float]:
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
