from collections import OrderedDict
import warnings

import flwr as fl
from flwr.common.typing import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(net, trainloader, epochs, dry_run=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        batch_num = 0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            if dry_run and batch_num > 1:
                break


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


# def main_numpy_client():
#     """Create model, load data, define Flower client, start Flower client."""

#     # Load model and data
#     net = Net().to(DEVICE)
#     trainloader, testloader, num_examples = load_data()

#     # Flower client
#     class CifarClient(fl.client.NumPyClient):
#         def get_properties(self, config):
#             return {"battery_level": 1.0}

#         def get_parameters(self):
#             return [val.cpu().numpy() for _, val in net.state_dict().items()]

#         def set_parameters(self, parameters):
#             params_dict = zip(net.state_dict().keys(), parameters)
#             state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#             net.load_state_dict(state_dict, strict=True)

#         def fit(self, parameters, config):
#             self.set_parameters(parameters)
#             train(net, trainloader, epochs=1, dry_run=True)
#             return self.get_parameters(), num_examples["trainset"], {}

#         def evaluate(self, parameters, config):
#             self.set_parameters(parameters)
#             loss, accuracy = test(net, testloader)
#             return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

#     # Start client
#     fl.client.start_numpy_client("[::]:8080", client=CifarClient())


def main_client():
    """Create model, load data, define Flower client, start Flower client."""

    # Load model and data
    net = Net().to(DEVICE)
    trainloader, testloader, num_examples = load_data()

    # Flower client
    class CifarClient(fl.client.Client):
        def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
            properties = {"battery_level": 1.0}
            return GetPropertiesRes(
                status=Status(code=Code.OK, message="Success"),
                properties=properties,
            )

        def get_parameters(self) -> GetParametersRes:
            parameters = [val.cpu().numpy() for _, val in net.state_dict().items()]
            parameters_proto = fl.common.weights_to_parameters(parameters)  # Serialize
            return GetParametersRes(parameters=parameters_proto)

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, ins: FitIns) -> FitRes:
            weights = fl.common.parameters_to_weights(ins.parameters)  # Deserialize
            self.set_parameters(weights)
            train(net, trainloader, epochs=1, dry_run=True)
            weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
            params = fl.common.weights_to_parameters(weights)  # Serialize
            return FitRes(
                parameters=params,
                num_examples=num_examples["trainset"],
                metrics={}
            )

        def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
            weights = fl.common.parameters_to_weights(ins.parameters)  # Deserialize
            self.set_parameters(weights)
            loss, accuracy = test(net, testloader)
            return EvaluateRes(
                loss=loss,
                num_examples=num_examples["testset"],
                metrics={"accuracy": float(accuracy)},
            )

    # Start client
    fl.client.start_client("[::]:8080", client=CifarClient())


if __name__ == "__main__":
    main_client()
