from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet34
from tqdm import tqdm

from FedPer.utils.model_manager import ModelManager
from FedPer.utils.model_split import ModelSplit


class ResNet(nn.Module):
    """ResNet model."""

    def __init__(
        self,
        num_head_layers: int = 1,
        num_classes: int = 10,
        device: str = "cpu",
        name: str = "resnet",
    ) -> None:
        super(ResNet, self).__init__()
        assert (
            num_head_layers > 0 and num_head_layers <= 17
        ), "num_head_layers must be greater than 0 and less than 16"
        self.num_head_layers = num_head_layers
        self.body = resnet34()

        def basic_block(in_planes, out_planes, stride_use=1):
            """Basic ResNet block."""
            return nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=(3, 3),
                    stride=(stride_use, stride_use),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_planes,
                    eps=1e-05,
                    momentum=0.01,
                    affine=True,
                    track_running_stats=True,
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_planes,
                    out_planes,
                    kernel_size=(3, 3),
                    stride=(stride_use, stride_use),
                    padding=(1, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(
                    out_planes,
                    eps=1e-05,
                    momentum=0.01,
                    affine=True,
                    track_running_stats=True,
                ),
            )

        # if only one head layer
        if self.num_head_layers == 1:
            self.head = self.body.fc
            self.body.fc = nn.Identity()
        elif self.num_head_layers == 2:
            self.head = nn.Sequential(
                basic_block(512, 512),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_classes),
            )
            # remove head layers from body
            self.body = nn.Sequential(*list(self.body.children())[:-2])
            body_layer4 = list(self.body.children())[-1]
            self.body = nn.Sequential(*list(self.body.children())[:-1])
            self.body.layer4 = nn.Sequential(*list(body_layer4.children())[:-1])
        else:
            raise NotImplementedError("Only 1 or 2 head layers supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print("Forwarding through ResNet model")
        if self.num_head_layers == 1:
            return self.head(F.relu(self.body(x)))
        elif self.num_head_layers == 2:
            # print("x.shape: ", x.shape)
            x = self.body(x)
            # x = F.relu(x)
            # x = x.view(x.size(0), 512, 1, 1)
            return self.head(x)
        else:
            raise NotImplementedError("Only 1 or 2 head layers supported")


class ResNetModelSplit(ModelSplit):
    """Concrete implementation of ModelSplit for models for node kind prediction in
    action flows \\ with Body/Head split.
    """

    def _get_model_parts(self, model: ResNet) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head

class ResNetModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_id: int,
        config: Dict[str, Any],
        trainloader: DataLoader,
        testloader: DataLoader,
        has_fixed_head: bool = False,
        client_save_path: str = None,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            has_fixed_head: Whether a fixed head should be created.
        """
        super().__init__(
            model_split_class=ResNetModelSplit,
            client_id=client_id,
            config=config,
            has_fixed_head=has_fixed_head,
        )
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config["device"]
        self.client_save_path = client_save_path

    def _create_model(self) -> nn.Module:
        """Return MobileNet-v1 model to be splitted into head and body."""
        try:
            return ResNet(
                num_head_layers=self.config["num_head_layers"],
                num_classes=self.config["num_classes"],
            ).to(self.device)
        except AttributeError:
            self.device = self.config["device"]
            return ResNet(
                num_head_layers=self.config["num_head_layers"],
                num_classes=self.config["num_classes"],
            ).to(self.device)

    def train(
        self,
        train_id: int,
        epochs: int = 1,
        tag: Optional[str] = None,
        fine_tuning: bool = False,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Method adapted from simple MobileNet-v1 (PyTorch) \
        https://github.com/wjc852456/pytorch-mobilenet-v1.

        Args:
            train_id: id of the train round.
            epochs: number of training epochs.
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                              (FedAvg, FedPer, FedRep, FedBABU or FedHybridAvgLGDual).
                              In the case of FedHybridAvgLGDual the tag also includes which part of the algorithm\
                                is being performed, either FedHybridAvgLGDual_FedAvg or FedHybridAvgLGDual_LG-FedAvg.
                <model_train_part> - indicates the part of the model that is being trained (full, body, head).
                This tag can be ignored if no difference in train behaviour is desired between federated algortihms.
            fine_tuning: whether the training performed is for model fine-tuning or not.

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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        correct, total, loss = 0, 0, 0.0
        # self.model.train()
        for _ in range(epochs):
            for images, labels in tqdm(self.trainloader):
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

    def test(self, test_id: int) -> Dict[str, float]:
        """Test the model maintained in self.model.

        Method adapted from simple CNN from Flower 'Quickstart PyTorch' \
        (https://flower.dev/docs/quickstart-pytorch.html).

        Args:
            test_id: id of the test round.

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
            for images, labels in tqdm(self.testloader):
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
