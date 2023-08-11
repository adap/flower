import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from torch import Tensor
from typing import Any, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from FedPer.utils.model_split import ModelSplit
from FedPer.utils.model_manager import ModelManager

from torchvision.models.resnet import resnet34

class ResNet(nn.Module):
    """Model adapted from simple MobileNet-v1 (PyTorch) \
        (https://github.com/wjc852456/pytorch-mobilenet-v1)."""

    def __init__(
            self, 
            num_head_layers : int = 1, 
            num_classes : int = 10, 
            device : str = 'cpu', 
            name : str = 'mobile'
        ) -> None:
        super(ResNet, self).__init__()
        assert num_head_layers > 0 and num_head_layers < 16, "num_head_layers must be greater than 0 and less than 16"
        self.body = None
        self.head = None

        # Create resnet34 model
        self.resnet = resnet34()

        # Self head is classifier and avgpool (initially)
        # body is the rest
        if num_head_layers == 1:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, num_classes)
            )
            self.body = nn.Sequential(*list(self.resnet.children())[:-2])
        else:
            # Add first 4 layers of resnet to body
            self.body = nn.Sequential(*list(self.resnet.children())[0:4])
            rest_to_add = list(self.resnet.children())[4:-2]
            num_body_layers = 16 - num_head_layers + 1 
            i = 0
            for j, layer in enumerate(rest_to_add):
                new_layer = []
                for n, block in enumerate(layer):
                    if i < num_body_layers:
                        new_layer.append(block)
                        i += 1
                        # Rest_to_add
                        to_add = layer[n+1:]
                        rest_to_add[j] = nn.Sequential(*to_add)
                        #print(layer[n+1:])
                    else:
                        break
                self.body.add_module('layer' + str(j), nn.Sequential(*new_layer))

                
            # Add rest of layers to head
            self.head = nn.Sequential(*rest_to_add)
            self.head.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
            self.head.add_module('flatten', nn.Flatten())
            self.head.add_module('classifier', nn.Linear(512, num_classes))       

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)

class ResNetModelSplit(ModelSplit):
    """Concrete implementation of ModelSplit for models for node kind prediction in action flows \
        with Body/Head split."""

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
            model_split_class=ResNetModelSplit,
            client_id=client_id,
            config=config,
            has_fixed_head=has_fixed_head
        )
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config['device']

    def _create_model(self) -> nn.Module:
        """Return MobileNet-v1 model to be splitted into head and body."""
        try:
            return ResNet().to(self.device)
        except AttributeError:
            self.device = self.config['device']
            return ResNet().to(self.device)

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