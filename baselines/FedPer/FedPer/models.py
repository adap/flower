"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

import time
import tqdm
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional, Dict, Tuple
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torchvision.models import resnet34

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MobileNet_v1(nn.Module):
    """ 
        MobileNet_v1 class. 
        As implemented in https://github.com/wjc852456/pytorch-mobilenet-v1. 
        This is a PyTorch implementation of MobileNet_v1.

        Includes a body and head if the model is split. 

        Args:
            split: whether to split the model into body and head.
            num_head_layers: number of layers in the head.
    """
    def __init__(self, split : bool = False, num_head_layers : int = 0) -> None:
        super(MobileNet_v1, self).__init__()

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
            'layer_16' : {'fc' : [1024, 1000]}
        }

        if split:
            self.body = MobileNet_v1_body(num_head_layers, ARCHITECTURE)
            self.head = MobileNet_v1_head(num_head_layers, ARCHITECTURE)
        else:
            NotImplementedError("MobileNet_v1 without split is not implemented yet.")

    def forward(self, x : Tensor) -> Tensor:
        x = self.body(x)
        x = self.head(x)
        return x
    
class MobileNet_v1_body(nn.Module):
    """ 
    Body of the MobileNet_v1 model, for which n layers at the end are removed. 
    
    Args:
        num_head_layers: number of layers in the head.
        architecture: architecture of the model.
    """
    def __init__(self, num_head_layers : int = 1, architecture : dict = None) -> None: 
        super(MobileNet_v1_body, self).__init__()
        assert num_head_layers >= 1, "Number of head layers must be at least 1."
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
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
        
        def avg_pool(value : int):
            return nn.AvgPool2d(value)
        
        def fc(inp, oup):
            return nn.Linear(inp, oup)
        
        self.model = nn.Sequential()
        for i in range(1, len(architecture) - num_head_layers + 1):
            for key, value in architecture[f'layer_{i}'].items():
                if key == 'conv_bn':
                    self.model.add_module(f'conv_bn_{i}', conv_bn(*value))
                elif key == 'conv_dw':
                    self.model.add_module(f'conv_dw_{i}', conv_dw(*value))
                elif key == 'avg_pool':
                    self.model.add_module(f'avg_pool_{i}', avg_pool(*value))
                elif key == 'fc':
                    self.model.add_module(f'fc_{i}', fc(*value))
                else:
                    raise NotImplementedError("Layer type not implemented.")
                
    def forward(self, x : Tensor) -> Tensor:
        x = self.model(x)
        return x

class MobileNet_v1_head(nn.Module):
    """ 
    MobileNet_v1 head, consists out of n layers that will be added to body of model. 
    
    Args:
        num_head_layers: number of layers in the head.
        architecture: architecture of the model.

    """

    def __init__(self, num_head_layers : int = 1, architecture : dict = None) -> None:
        super(MobileNet_v1_head, self).__init__()
        assert num_head_layers >= 1, "Number of head layers must be at least 1."
        self.num_head_layers = num_head_layers
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
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
        
        def avg_pool(value : int):
            return nn.AvgPool2d(value)
        
        def fc(inp, oup):
            return nn.Linear(inp, oup)
        
        self.model = nn.Sequential()
        for i in range(len(architecture) - num_head_layers + 1, len(architecture) + 1):
            for key, value in architecture[f'layer_{i}'].items():
                if key == 'conv_bn':
                    self.model.add_module(f'conv_bn_{i}', conv_bn(*value))
                elif key == 'conv_dw':
                    self.model.add_module(f'conv_dw_{i}', conv_dw(*value))
                elif key == 'avg_pool':
                    self.model.add_module(f'avg_pool_{i}', avg_pool(*value))
                elif key == 'fc':
                    self.model.add_module(f'fc_{i}', fc(*value))
                else:
                    raise NotImplementedError("Layer type not implemented.")
    
    def forward(self, x : Tensor) -> Tensor:
        if self.num_head_layers != 1:
            x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

class ModelSplit(nn.Module):
    """Class for splitting a model into body and head. Optionally, a fixed head can also be created."""

    def __init__(
            self,
            model: nn.Module,
            has_fixed_head: bool = False
    ):
        """
        Initialize ModelSplit attributes. A call is made to the _setup_model_parts method.

        Args:
            model: dict containing the vocab sizes of the input attributes.
            has_fixed_head: whether the model should contain a fixed_head.
        """
        super().__init__()

        self._body, self._head = self._get_model_parts(model)
        self._fixed_head = copy.deepcopy(self.head) if has_fixed_head else None
        self._use_fixed_head = False

    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """
        Return the body and head of the model.

        Args:
            model: model to be split into head and body

        Returns:
            Tuple where the first element is the body of the model and the second is the head.
        """
        return model.body, model.head

    @property
    def body(self) -> nn.Module:
        """Return model body."""
        return self._body

    @body.setter
    def body(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """
        Set model body.

        Args:
            state_dict: dictionary of the state to set the model body to.
        """
        self.body.load_state_dict(state_dict, strict=True)

    @property
    def head(self) -> nn.Module:
        """Return model head."""
        return self._head

    @head.setter
    def head(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """
        Set model head.

        Args:
            state_dict: dictionary of the state to set the model head to.
        """
        self.head.load_state_dict(state_dict, strict=True)

    @property
    def fixed_head(self) -> Optional[nn.Module]:
        """Return model fixed_head."""
        return self._fixed_head

    @fixed_head.setter
    def fixed_head(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """
        Set model fixed_head.

        Args:
            state_dict: dictionary of the state to set the model fixed head to.
        """
        if self._fixed_head is None:
            # When the fixed_head was not initialized
            return
        self._fixed_head.load_state_dict(state_dict, strict=True)
        self.disable_fixed_head()

    def get_parameters(self) -> List[np.ndarray]:
        """
        Get model parameters (without fixed head).

        Returns:
            Body and head parameters
        """
        return [
            val.cpu().numpy() for val in [
                *self.body.state_dict().values(), *self.head.state_dict().values()
            ]
        ]

    def set_parameters(self, state_dict: Dict[str, Tensor]) -> None:
        """
        Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        # Copy to maintain the order of the parameters and add the missing parameters to the state_dict
        ordered_state_dict = OrderedDict(self.state_dict().copy())
        # Update with the values of the state_dict
        ordered_state_dict.update({k: v for k, v in state_dict.items()})
        self.load_state_dict(ordered_state_dict, strict=True)

    def enable_head(self) -> None:
        """Enable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = True

    def enable_body(self) -> None:
        """Enable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = True

    def disable_head(self) -> None:
        """Disable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = False

    def disable_body(self) -> None:
        """Disable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = False

    def disable_fixed_head(self) -> None:
        """Disable gradient tracking for the fixed head parameters."""
        if self._fixed_head is None:
            return
        for param in self._fixed_head.parameters():
            param.requires_grad = False

    def use_fixed_head(self, use_fixed_head: bool) -> None:
        """
        Set whether the fixed head should be used for forward.

        Args:
            use_fixed_head: boolean indicating whether to use the fixed head or not.
        """
        self._use_fixed_head = use_fixed_head

    def forward(self, inputs: Any) -> Any:
        """Forward inputs through the body and the head (or fixed head)."""
        x = self.body(inputs)
        if self._use_fixed_head and self.fixed_head is not None:
            return self.fixed_head(x)
        return self.head(x)
    
class ModelManager():
    """Manager for models with Body/Head split."""

    def __init__(
            self,
            client_id: int,
            config: Dict[str, Any],
            model_split_class: Type[ModelSplit],
            has_fixed_head: bool = False
    ):
        """
        Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            model_split_class: Class to be used to split the model into body and head\
                (concrete implementation of ModelSplit).
            has_fixed_head: Whether a fixed head should be created.
        """
        super().__init__()

        self.client_id = client_id
        self.config = config
        self._model = model_split_class(self._create_model(), has_fixed_head)

    def _create_model(self) -> nn.Module:
        """Return model to be splitted into head and body."""
        return MobileNet_v1(split=True, num_head_layers=1)

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


def train(
    self,
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> None:
        """ Trains the model on the given data.
        Parameters
        ----------
        net : nn.Module
            The model to be trained.
        trainloader : DataLoader
            The data to train the model on.
        device : torch.device
            The device to use for training, either 'cpu' or 'cuda'.
        epochs : int
            The number of epochs to train the model for.
        learning_rate : float
            The learning rate to use for training.

        Returns
        -------
        None
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=5e-4)
        net.train()
        for _ in range(epochs):
            net = _train_one_epoch(
                net=net,
                trainloader=trainloader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
            )

def _train_one_epoch(
        net: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        criterion: torch.nn.CrossEntropyLoss,
        optimizer: torch.optim.SGD,
) -> nn.Module:
    """ Trains the model on the given data for one epoch.

    Parameters
    ----------
    net : nn.Module
        The model to be trained.
    trainloader : DataLoader
        The data to train the model on.
    device : torch.device
        The device to use for training, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training.
    optimizer : torch.optim.SGD
        The optimizer to use for training.

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return net

def test(
        net: nn.Module,
        testloader: DataLoader,
        device: torch.device,
) -> Tuple[float, float]:
    """ Evaluates the model on the given data.
    
    Parameters
    ----------
    net : nn.Module
        The model to be evaluated.
    testloader : DataLoader
        The data to evaluate the model on.
    device : torch.device
        The device to use for evaluation, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and accuracy of the model on the given data.
    """

    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if len(testloader) == 0:
        raise ValueError("testloader is empty, please provide a valid testloader")
    loss /= len(testloader.dataset)
    accuracy = 100 * correct / total

    return loss, accuracy
