import json

from copy import deepcopy
from typing import Dict, List, Optional, Type, Tuple
from pathlib import Path
from collections import OrderedDict

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.utils.data import DataLoader, Subset

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()

class DecoupledModel(nn.Module):
    """ Base class for decoupled models."""
    def __init__(
            self, 
            num_classes : int = 10,
            # num_classifier_layer : int = 1,
            device : str = "cpu",
            name : str = "resnet"
        ):
        super(DecoupledModel, self).__init__()
        """ Base class for decoupled models."""
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: List[nn.Module] = []

        self.num_classes = num_classes
        self.device = device
        self.name = name

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.clone().detach())

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: torch.Tensor):
        if self.num_classifier_layers == 1:
            return self.classifier(F.relu(self.base(x)))
        elif self.num_classifier_layers == 2:
            x = self.base(x)
            x = F.relu(x)
            x = x.view(x.size(0), 512, 1, 1)
            x = self.classifier(x)
            return x

    def get_final_features(self, x: torch.Tensor, detach=True) -> torch.Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: torch.Tensor) -> Optional[List[torch.Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list

class MobileNetV2(DecoupledModel):
    def __init__(self, dataset):
        super(MobileNetV2, self).__init__()
        config = {
            "cifar10": 10,
            "cifar100": 100,
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
        )
        self.classifier = nn.Linear(
            self.base.classifier[1].in_features, config[dataset]
        )

        self.base.classifier[1] = nn.Identity()

class ResNet18(DecoupledModel):
    def __init__(
            self, 
            dataset: str = "cifar10",
            num_classes : int = 10,
            num_classifier_layers : int = 1,
            device : str = "cpu",
            name : str = "resnet"
        ):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.num_classifier_layers = num_classifier_layers
        self.device = device
        self.name = name

        config = {
            "cifar10": 10,
            "cifar100": 100,
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        self.classifier = nn.Linear(self.base.fc.in_features, config[dataset])
        self.base.fc = nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().forward(x)

    def get_all_features(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)

class ResNet34(DecoupledModel):
    def __init__(
            self, 
            dataset: str = "cifar10",
            num_classes : int = 10,
            num_classifier_layers : int = 1,
            device : str = "cpu",
            name : str = "resnet"
        ):
        super(ResNet34, self).__init__()
        self.num_classes = num_classes
        self.num_classifier_layers = num_classifier_layers
        self.device = device
        self.name = name

        config = {
            "cifar10": 10,
            "cifar100": 100,
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        self.base = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        def basic_block(in_planes, planes, stride_use=1):
            """Basic ResNet building block."""
            return nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=(3,3), stride=(stride_use, stride_use), padding=(1,1), bias=False),
                nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(planes, planes, kernel_size=(3,3), stride=(stride_use, stride_use), padding=(1,1), bias=False),
                nn.BatchNorm2d(planes, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )

        def downsample(in_planes, planes, stride=1):
            """Downsample for ResNet."""
            return nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        if num_classifier_layers == 1:
            self.classifier = nn.Linear(self.base.fc.in_features, config[dataset])
            self.base.fc = nn.Identity()
        elif num_classifier_layers == 2:
            self.classifier = nn.Sequential(
                basic_block(self.base.fc.in_features, self.base.fc.in_features),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(self.base.fc.in_features, config[dataset])
            )
            self.base.fc = nn.Identity()
            self.base.avgpool = nn.Identity()
            self.base.layer4[-1] = nn.Identity()

    def forward(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().forward(x)

    def get_all_features(self, x):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_all_features(x)

    def get_final_features(self, x, detach=True):
        if x.shape[1] == 1:
            x = torch.expand_copy(x, (x.shape[0], 3, *x.shape[2:]))
        return super().get_final_features(x, detach)

MODEL_DICT: Dict[str, Type[DecoupledModel]] = {
    "mobile": MobileNetV2,
    "res18": ResNet18,
}

def train(
    net: DecoupledModel,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
    """Train the network on the training set.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    epochs : int
        The number of epochs the model should be trained for.
    learning_rate : float
        The learning rate for the SGD optimizer.
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    net.train()
    for _ in range(epochs):
        net = _training_loop(net, trainloader, device, criterion, optimizer)

def _training_loop(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
) -> nn.Module:
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    for images, labels in tqdm(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()
    return net

def test(
    net: DecoupledModel, 
    testloader: DataLoader, 
    device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    # print("Accuracy of the client on its test images: %d %%" % (100 * accuracy))
    return loss, accuracy
