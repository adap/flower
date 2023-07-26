import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from FedPer.new_utils import ModelManager, ModelSplit
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CNNNetBody(nn.Module):
    """Model adapted from simple CNN from Flower 'Quickstart PyTorch' \
        (https://flower.dev/docs/quickstart-pytorch.html)."""

    def __init__(self) -> None:
        super(CNNNetBody, self).__init__()
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

class CNNNet(nn.Module):
    """Model adapted from simple CNN from Flower 'Quickstart PyTorch' \
        (https://flower.dev/docs/quickstart-pytorch.html)."""

    def __init__(self) -> None:
        super(CNNNet, self).__init__()
        self.body = CNNNetBody()
        self.head = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)
    
class CNNModelSplit(ModelSplit):
    """Concrete implementation of ModelSplit for models for node kind prediction in action flows \
        with Body/Head split."""

    def _get_model_parts(self, model: CNNNet) -> Tuple[nn.Module, nn.Module]:
        return model.body, model.head

class CNNModelManager(ModelManager):
    """Manager for models with Body/Head split."""

    def __init__(
            self,
            client_id: int,
            config: Dict[str, Any],
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
            model_split_class=CNNModelSplit,
            client_id=client_id,
            config=config,
            has_fixed_head=has_fixed_head
        )

        self.trainloader, self.testloader = self.load_data()

    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """
        Load CIFAR-10 (training and test set).

        Method adapted from simple CNN from Flower 'Quickstart PyTorch' \
        (https://flower.dev/docs/quickstart-pytorch.html).
        """
        trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = CIFAR10("./data", train=True, download=True, transform=trf)
        testset = CIFAR10("./data", train=False, download=True, transform=trf)
        return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

    def _create_model(self) -> nn.Module:
        """Return CNN model to be splitted into head and body."""
        return CNNNet()

    def train(
        self,
        train_id: int,
        epochs: int = 1,
        tag: Optional[str] = None,
        fine_tuning: bool = False
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Train the model maintained in self.model.

        Method adapted from simple CNN from Flower 'Quickstart PyTorch' \
        (https://flower.dev/docs/quickstart-pytorch.html).

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
                criterion(self.model(images.to(DEVICE)), labels.to(DEVICE)).backward()
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
                outputs = self.model(images.to(DEVICE))
                labels = labels.to(DEVICE)
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