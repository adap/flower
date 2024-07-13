"""ResNet model, model manager and split."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet34

from fedper.models import ModelManager, ModelSplit


def conv3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Basic block for ResNet."""

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward inputs through the block."""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(
        self,
        num_head_layers: int = 1,
        num_classes: int = 10,
    ) -> None:
        super(ResNet, self).__init__()
        assert (
            num_head_layers > 0 and num_head_layers <= 17
        ), "num_head_layers must be greater than 0 and less than 16"

        self.num_head_layers = num_head_layers
        self.body = resnet34()

        # if only one head layer
        if self.num_head_layers == 1:
            self.head = self.body.fc
            self.body.fc = nn.Identity()
        elif self.num_head_layers == 2:
            self.head = nn.Sequential(
                BasicBlock(512, 512),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )
            # remove head layers from body
            self.body = nn.Sequential(*list(self.body.children())[:-2])
            body_layer4 = list(self.body.children())[-1]
            self.body = nn.Sequential(*list(self.body.children())[:-1])
            self.body.layer4 = nn.Sequential(*list(body_layer4.children())[:-1])
        elif self.num_head_layers == 3:
            self.head = nn.Sequential(
                BasicBlock(512, 512),
                BasicBlock(512, 512),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )
            # remove head layers from body
            self.body = nn.Sequential(*list(self.body.children())[:-2])
            body_layer4 = list(self.body.children())[-1]
            self.body = nn.Sequential(*list(self.body.children())[:-1])
            self.body.layer4 = nn.Sequential(*list(body_layer4.children())[:-2])
        else:
            raise NotImplementedError("Only 1 or 2 head layers supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward inputs through the model."""
        print("Forwarding through ResNet model")
        x = self.body(x)
        return self.head(x)


class ResNetModelSplit(ModelSplit):
    """Split ResNet model into body and head."""

    def _get_model_parts(self, model: ResNet) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


class ResNetModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_save_path: Optional[str],
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        learning_rate: float = 0.01,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_save_path: Path to save the client state.
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            trainloader: DataLoader containing the train data.
            testloader: DataLoader containing the test data.
            learning_rate: Learning rate for the optimizer.
        """
        super().__init__(
            model_split_class=ResNetModelSplit,
            client_id=client_id,
            config=config,
        )
        self.client_save_path = client_save_path
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config["server_device"]
        self.learning_rate = learning_rate

    def _create_model(self) -> nn.Module:
        """Return MobileNet-v1 model to be splitted into head and body."""
        try:
            return ResNet(
                num_head_layers=self.config["model"]["num_head_layers"],
                num_classes=self.config["model"]["num_classes"],
            ).to(self.device)
        except AttributeError:
            self.device = self.config["server_device"]
            return ResNet(
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
        """Test the model maintained in self.model."""
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
