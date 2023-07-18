"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""

from typing import List, Tuple

import time
import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional, Dict
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.nn.parameter import Parameter
from torchvision.models import resnet34


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
        
        def avg_pool():
            return nn.AvgPool2d((1, 1))
        
        def fc(inp, oup):
            return nn.Linear(inp, oup)
        
        



class ModelSplit(ABC, nn.Module):
    """Abstract class for splitting a model into body and head. Optionally, a fixed head can also be created."""

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

    @abstractmethod
    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """
        Return the body and head of the model.

        Args:
            model: model to be split into head and body

        Returns:
            Tuple where the first element is the body of the model and the second is the head.
        """
        pass

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
        return [val.cpu().numpy() for val in [*self.body.state_dict().values(), *self.head.state_dict().values()]]

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
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
        global_params = [val.detach().clone() for val in net.parameters()]
        net.train()
        for _ in range(epochs):
            net = _train_one_epoch(
                net=net,
                global_params=global_params,
                trainloader=trainloader,
                device=device,
                criterion=criterion,
                optimizer=optimizer,
            )

def _train_one_epoch(
        net: nn.Module,
        global_params: List[Parameter],
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
    global_params : List[Parameter]
        The global parameters to be used for training.
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



def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None, 
        has_bn = True,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        if has_bn:
            self.bn2 = norm_layer(planes)
        else:
            self.bn2 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        if has_bn:
            self.bn3 = norm_layer(planes)
        else:
            self.bn3 = nn.Identity()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        has_bn = True,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        if has_bn:
            self.bn1 = norm_layer(width)
        else:
            self.bn1 = nn.Identity()
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        if has_bn:
            self.bn2 = norm_layer(width)
        else:
            self.bn2 = nn.Identity()
        self.conv3 = conv1x1(width, planes * self.expansion)
        if has_bn:
            self.bn3 = norm_layer(planes * self.expansion)
        else:
            self.bn3 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet(nn.Module):

    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        features: List[int] = [64, 128, 256, 512],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None, 
        has_bn = True,
        bn_block_num = 4, 
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        if has_bn:
            self.bn1 = norm_layer(self.inplanes)
        else:
            self.bn1 = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layers = []
        self.layers.extend(self._make_layer(block, 64, layers[0], has_bn=has_bn and (bn_block_num > 0)))
        for num in range(1, len(layers)):
            self.layers.extend(self._make_layer(block, features[num], layers[num], stride=2,
                                       dilate=replace_stride_with_dilation[num-1], 
                                       has_bn=has_bn and (num < bn_block_num)))

        for i, layer in enumerate(self.layers):
            setattr(self, f'layer_{i}', layer)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(features[len(layers)-1] * block.expansion, num_classes)

        # self.fc = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)), 
        #     nn.Flatten(), 
        #     nn.Linear(features[len(layers)-1] * block.expansion, num_classes)
        # )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, has_bn=True) -> List:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if has_bn:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    nn.Identity(),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, has_bn))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, has_bn=has_bn))

        return layers

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i in range(len(self.layers)):
            layer = getattr(self, f'layer_{i}')
            x = layer(x)

        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
def resnet34(**kwargs: Any) -> ResNet: 
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
