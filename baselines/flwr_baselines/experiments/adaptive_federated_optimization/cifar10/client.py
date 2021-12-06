import sys
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union

import flwr as fl
import numpy as np
import ray
import torch
from flwr.common.parameter import (
    Parameters,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.typing import Scalar
from PIL import Image
from torch import Tensor
from torch.nn import GroupNorm, Module
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.models import ResNet, resnet18
from torchvision.transforms import Compose, Normalize, RandomHorizontalFlip, ToTensor

transforms_test = Compose(
    [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
transforms_train = Compose([RandomHorizontalFlip(), transforms_test])


def get_model(num_classes: int = 10) -> Module:
    model: ResNet = resnet18(
        norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes
    )
    return model


class ClientDataset(Dataset):
    def __init__(self, path_to_data: Path, transform: Compose = None):
        super().__init__()
        self.transform = transform
        self.X, self.Y = torch.load(path_to_data)

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: Union[int, Tensor]) -> Tuple[Tensor, int]:
        # if torch.is_tensor(idx):
        #    idx = idx.tolist()
        x = Image.fromarray(self.X[idx])
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def train(
    net: Module,
    trainloader: DataLoader,
    epochs: int,
    device: str,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
) -> None:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net: Module, testloader: DataLoader, device: str) -> Tuple[float, float]:
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


class CifarRayClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir: Path):
        self.cid = cid
        self.fed_dir = fed_dir
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, net: Optional[Module] = None) -> Weights:
        if not net:
            net = get_model(Parameters)
        weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
        return weights

    def get_properties(self, ins: Dict[str, Scalar]) -> Dict[str, Scalar]:
        return self.properties

    def fit(
        self, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Tuple[Parameters, int, Dict[str, Scalar]]:
        net = self.set_parameters(parameters)
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainset = ClientDataset(
            path_to_data=Path(self.fed_dir) / f"{self.cid}" / "train.pt",
            transform=transforms_train,
        )
        net.to(self.device)

        # train
        trainloader = DataLoader(
           trainset, batch_size=int(config["batch_size"]), num_workers=num_workers
        )
        train(net, trainloader, epochs=int(config["epochs"]), device=self.device)

        # return local model and statistics
        return self.get_parameters(net), len(trainloader.dataset), {}

    def evaluate(
        self, parameters: Parameters, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, float]]:
        net = self.set_parameters(parameters)
        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        validationset = ClientDataset(
            path_to_data=Path(self.fed_dir) / self.cid / "test.pt"
        )
        valloader = DataLoader(validationset, batch_size=50, num_workers=num_workers)

        # send model to device
        net.to(self.device)

        # evaluate
        loss, accuracy = test(net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
    
    def set_parameters(self, parameters):
        net = get_model()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net
    


def get_eval_fn(
    testset: CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, Dict[str, float]]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        weights: fl.common.Weights,
    ) -> Optional[Tuple[float, Dict[str, float]]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = get_model()
        state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(np.state_dict().keys(), weights)
        })
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(net, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
