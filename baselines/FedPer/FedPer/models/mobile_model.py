import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from FedPer.utils.model_split import ModelSplit
from FedPer.utils.model_manager import ModelManager

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MobileNetBody(nn.Module):
    """Model adapted from simple CNN from Flower 'Quickstart PyTorch' \
        (https://flower.dev/docs/quickstart-pytorch.html)."""

    def __init__(self) -> None:
        super(MobileNetBody, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class MobileNet(nn.Module):
    """Model adapted from simple MobileNet-v1 (PyTorch) \
        (https://github.com/wjc852456/pytorch-mobilenet-v1)."""

    def __init__(self) -> None:
        super(MobileNet, self).__init__()

        ARCHITECTURE = {
            'layer_1' : {'conv_bn' : [3, 32, 2]},
            'layer_2' : {'conv_dw' : [32, 64, 1]},
            'layer_3' : {'conv_dw' : [64, 128, 2]},
            'layer_4' : {'conv_dw' : [128, 128, 1]},
            'layer_5' : {'conv_dw' : [128, 256, 2]},
            'layer_6' : {'conv_dw' : [256, 256, 1]},
            'layer_7' : {'conv_dw' : [256, 512, 2]},
            'layer_8' : {'conv_dw' : [512, 512, 1]},
            'layer_9' : {'conv_dw' : [512, 512, 1]},
            'layer_10' : {'conv_dw' : [512, 512, 1]},
            'layer_11' : {'conv_dw' : [512, 512, 1]},
            'layer_12' : {'conv_dw' : [512, 512, 1]},
            'layer_13' : {'conv_dw' : [512, 1024, 2]},
            'layer_14' : {'conv_dw' : [1024, 1024, 1]},
            'layer_15' : {'avg_pool' : [7]},
            'layer_16' : {'fc' : [1024, num_classes]}
        }

        self.body = MobileNetBody()
        self.head = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)
    
    def set_config(self, config : dict) -> None:
        """ 
            Set configuration file. 

            Parameters: 
                config: Dictionary with configurations. 
        """
        self.config = config

class MobileNetModelSplit(ModelSplit):
    """Concrete implementation of ModelSplit for models for node kind prediction in action flows \
        with Body/Head split."""

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
            has_fixed_head: bool = False
    ):
        """
        Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            has_fixed_head: Whether a fixed head should be created.
        """
        super().__init__(
            model_split_class=MobileNetModelSplit,
            client_id=client_id,
            config=config,
            has_fixed_head=has_fixed_head
        )
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config['device']

    def _create_model(self) -> nn.Module:
        """Return MobileNet-v1 model to be splitted into head and body."""
        try:
            return MobileNet().to(self.device)
        except AttributeError:
            self.device = self.config['device']
            return MobileNet().to(self.device)

    def train(
        self,
        train_id: int,
        epochs: int = 1,
        tag: Optional[str] = None,
        fine_tuning: bool = False
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Train the model maintained in self.model.

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

        Returns:
            Dict containing the train metrics.
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        for _ in range(epochs):
            for images, labels in tqdm(self.trainloader):
                optimizer.zero_grad()
                criterion(self.model(images.to(self.device)), labels.to(self.device)).backward()
                optimizer.step()
        return {}

    def test(self, test_id: int) -> Dict[str, float]:
        """
        Test the model maintained in self.model.

        Method adapted from simple CNN from Flower 'Quickstart PyTorch' \
        (https://flower.dev/docs/quickstart-pytorch.html).

        Args:
            test_id: id of the test round.

        Returns:
            Dict containing the test metrics.
        """
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in tqdm(self.testloader):
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        return {"loss": loss / len(self.testloader.dataset), "accuracy": correct / total}

    def train_dataset_size(self) -> int:
        """Return train data set size."""
        return len(self.trainloader)

    def test_dataset_size(self) -> int:
        """Return test data set size."""
        return len(self.testloader)

    def total_dataset_size(self) -> int:
        """Return total data set size."""
        return len(self.trainloader) + len(self.testloader)