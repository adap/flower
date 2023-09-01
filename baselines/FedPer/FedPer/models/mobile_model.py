from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from FedPer.utils.model_manager import ModelManager
from FedPer.utils.model_split import ModelSplit

# Set model architecture
ARCHITECTURE = {
    # 'layer_1' : {'conv_bn' : [3, 32, 2]},
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
    #'layer_15' : {'avg_pool' : [7]},
    #'layer_16' : {'fc' : [1024, num_classes]}
}


class MobileNet(nn.Module):
    """Model adapted from simple MobileNet-v1 (PyTorch) \
    (https://github.com/wjc852456/pytorch-mobilenet-v1).
    """

    def __init__(
        self,
        num_head_layers: int = 1,
        num_classes: int = 10,
        device: str = "cpu",
        name: str = "mobile",
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
                nn.AvgPool2d([7]), 
                nn.Flatten(),
                nn.Linear(1024, num_classes))
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
        x = self.body(x)
        return self.head(x)


class MobileNetModelSplit(ModelSplit):
    """Concrete implementation of ModelSplit for models for node kind prediction in
    action flows \\ with Body/Head split.
    """
    
    def _get_model_parts(self, model: MobileNet) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head


class MobileNetModelManager(ModelManager):
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
            model_split_class=MobileNetModelSplit,
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
            return MobileNet(
                num_head_layers=self.config["num_head_layers"],
                num_classes=self.config["num_classes"],
            ).to(self.device)
        except AttributeError:
            self.device = self.config["device"]
            return MobileNet(
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
