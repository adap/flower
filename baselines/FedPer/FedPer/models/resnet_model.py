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

class ResNetHead(nn.Module):
    """ 
    MobileNet_v1 head, consists out of n layers that will be added to body of model. 
    
    Args:
        num_head_layers: number of layers in the head.
        num_classes: number of classes (outputs)
    """
    def __init__(
            self, 
            num_head_layers : int = 1, 
            num_classes : int = 10, 
            rest_to_add : list = None
        ) -> None:
        super(ResNetHead, self).__init__()

        self.rest_to_add = rest_to_add
        
        # if only one head layer
        if num_head_layers == 1:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        else:
            assert self.rest_to_add is not None
            self.rest_to_add = [i for i in self.rest_to_add if i is not None]
                
            # Add rest of layers to head
            self.head = nn.Sequential(*rest_to_add) 
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rest_to_add is not None:
            x = self.head(x)
        x = self.avgpool(x)
        # x = nn.Flatten(x, 1)
        x = torch.flatten(x,1)
        return self.fc(x)

class ResNetBody(nn.Module):
    """ 
    ResNet Body, consists out of n layers that will be added to body of model. 
    
    Args:
        num_head_layers: number of layers in the head.
        num_classes: number of classes (outputs)
    """
    def __init__(
            self, 
            num_head_layers : int = 1, 
            num_classes : int = 10, 
        ) -> None:
        super(ResNetBody, self).__init__()
        self.num_head_layers = num_head_layers
        self.body = resnet34()

        def basic_block(in_planes, out_planes, stride=1):
            """Basic ResNet block."""
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_planes)
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
                nn.Linear(512, num_classes)
            )
            self.body.fc = nn.Identity()
            self.body.avgpool = nn.Identity()
            self.body.layer4[-1] = nn.Identity()
        else: 
            raise NotImplementedError("Only 1 or 2 head layers supported")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_head_layers == 1:
            return self.body(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            return self.body(x)

    def get_rest_to_add(self,) -> list:
        """ Return list of rest to add blocks (for head). """
        return self.rest_to_add

class ResNet(nn.Module):
    """ ResNet model. """

    def __init__(
            self, 
            num_head_layers : int = 1, 
            num_classes : int = 10, 
            device : str = 'cpu', 
            name : str = 'resnet'
        ) -> None:
        super(ResNet, self).__init__()
        assert num_head_layers > 0 and num_head_layers <= 17, "num_head_layers must be greater than 0 and less than 16"
        self.num_head_layers = num_head_layers
        self.body = resnet34()

        def basic_block(in_planes, out_planes, stride_use=1):
            """Basic ResNet block."""
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=(3,3), stride=(stride_use, stride_use), padding=(1,1), bias=False),
                nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, kernel_size=(3,3), stride=(stride_use, stride_use), padding=(1,1), bias=False),
                nn.BatchNorm2d(out_planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
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
                nn.Linear(512, num_classes)
            )
            # remove head layers from body
            self.body = nn.Sequential(*list(self.body.children())[:-2])
            body_layer4 = list(self.body.children())[-1]
            self.body = nn.Sequential(*list(self.body.children())[:-1])
            self.body.layer4 = nn.Sequential(*list(body_layer4.children())[:-1])
        else: 
            raise NotImplementedError("Only 1 or 2 head layers supported")
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_head_layers == 1:
            return self.head(F.relu(self.body(x)))
        elif self.num_head_layers == 2:
            x = self.body(x)
            # x = F.relu(x)
            # x = x.view(x.size(0), 512, 1, 1)
            return self.head(x)
        else:
            raise NotImplementedError("Only 1 or 2 head layers supported")

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
            return ResNet(
                num_head_layers=self.config['num_head_layers'],
                num_classes=self.config['num_classes'],
            ).to(self.device)
        except AttributeError:
            self.device = self.config['device']
            return ResNet(
                num_head_layers=self.config['num_head_layers'],
                num_classes=self.config['num_classes'],
            ).to(self.device)

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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        correct, total, loss = 0, 0, 0.0
        #self.model.train()
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
        return {"loss": loss.item(), "accuracy": correct / total}


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
        #self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(self.testloader):
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        print("Test Accuracy: {:.4f}".format(correct / total))
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