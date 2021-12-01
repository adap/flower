import flwr as fl
import numpy as np
import ray
import torch

from collections import OrderedDict
from pathlib import Path
from PIL import Image
from torch.nn import GroupNorm
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from torchvision.transforms import Compose, Normalize, ToTensor, RandomHorizontalFlip
from typing import Dict, Callable, Optional, Tuple
from flwr.common.typing import Scalar

transforms_test = Compose(
    [ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
)
transforms_train = Compose([RandomHorizontalFlip(), transforms_test])


def get_resnet18_gn(num_classes: int = 10):
    model = resnet18(norm_layer=lambda x: GroupNorm(2, x), num_classes=num_classes)
    return model


class CIFARClientDataset(Dataset):
    def __init__(self, path_to_file: Path, transform=None):
        super().__init__()
        self.transform = transform
        self.X, self.Y = torch.load(path_to_file)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = Image.fromarray(self.X[idx])
        y = self.Y[idx]

        if self.transform:
            x = self.transform(x)
        return x, y


def train(net, trainloader, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader, device: str):
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

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, net=None):
        if not net:
            net = get_resnet18_gn()
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    # def get_properties(self, ins: PropertiesIns) -> PropertiesRes:
    def get_properties(self, ins):
        return self.properties

    def set_parameters(self, parameters):
        net = get_resnet18_gn()
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        return net

    def fit(self, parameters, config):
        net = self.set_parameters(parameters)
        num_workers = len(ray.worker.get_resource_ids()["CPU"])

        trainset = CIFARClientDataset(
            path_to_file=Path(self.fed_dir) / f"{self.cid}" / "train.pt",
            transform=transforms_train,
        )
        trainloader = DataLoader(
            trainset, batch_size=int(config["batch_size"]), num_workers=num_workers
        )
        # trainloader = DataLoader(trainset, batch_size=int(config["batch_size"]))
        net.to(self.device)

        # train
        train(net, trainloader, epochs=int(config["epochs"]), device=self.device)

        # return local model and statistics
        return self.get_parameters(net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        validationset = CIFARClientDataset(
            path_dir=Path(self.fed_dir) / self.cid / "test.pt"
        )
        valloader = DataLoader(validationset, batch_size=50, num_workers=num_workers)

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = get_resnet18_gn()
        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
