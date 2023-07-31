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

        self.body = ResNetBody(num_head_layers=num_head_layers, architecture=ARCHITECTURE)
        self.head = ResNetHead(num_head_layers=num_head_layers, architecture=ARCHITECTURE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.body(x)
        return self.head(x)
    
class ResNetBody(nn.Module):
    """ 
    Body of the MobileNet_v1 model, for which n layers at the end are removed. 
    
    Args:
        num_head_layers: number of layers in the head.
        architecture: architecture of the model.
    """
    def __init__(self, num_head_layers : int = 1, architecture : dict = None) -> None: 
        super(ResNetBody, self).__init__()
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
        for i in range(len(self.model)):
            x = self.model[i](x)
            if isinstance(self.model[i], nn.AvgPool2d):
                x = x.view(-1, 1024)
        return x

class ResNetHead(nn.Module):
    """ 
    MobileNet_v1 head, consists out of n layers that will be added to body of model. 
    
    Args:
        num_head_layers: number of layers in the head.
        architecture: architecture of the model.

    """

    def __init__(self, num_head_layers : int = 1, architecture : dict = None) -> None:
        super(ResNetHead, self).__init__()
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
        for i in range(len(self.model)):
            x = self.model[i](x)
            if isinstance(self.model[i], nn.AvgPool2d):
                x = x.view(-1, 1024)
        return x

class ResNetModelSplit(ModelSplit):
    """Concrete implementation of ModelSplit for models for node kind prediction in action flows \
        with Body/Head split."""

    def _get_model_parts(self, model: ResNet) -> Tuple[nn.Module, nn.Module]:
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