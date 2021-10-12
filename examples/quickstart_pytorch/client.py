from collections import OrderedDict
import warnings

import flwr as fl
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


def train(net, optimizer, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


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
    return trainloader, testloader

def get_sgd_optimizer_state(optimizer: torch.optim.SGD):
    """Return params from optimizer state"""
    params = []
    state_dict = optimizer.state_dict() # Has two keys: state, param_groups
    state = state_dict["state"]
    for _, val in state.items():
        tensor = val["momentum_buffer"]
        params.append(tensor.numpy())
    return params

def set_sgd_optimizer_state(optimizer: torch.optim.SGD, params):
    """Set params in optimizer state"""
    state_dict = optimizer.state_dict() # Has two keys: state, param_groups
    state = state_dict["state"]
    for idx, param in enumerate(params):
        state[idx] = { "momentum_buffer": torch.from_numpy(param) }
    optimizer.load_state_dict(state_dict)

# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def main():
    """Create model, load data, define Flower client, start Flower client."""

    # Load model
    net = Net().to(DEVICE)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    num_parameter_groups = len(list(net.parameters()))

    # Load data (CIFAR-10)
    trainloader, testloader = load_data()

    # Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_properties(self):
            return {}

        def get_parameters(self):
            net_params = [val.cpu().numpy() for _, val in net.state_dict().items()]
            opt_params = get_sgd_optimizer_state(optimizer)
            params = net_params + opt_params
            return params

        def set_parameters(self, parameters):
            net_params = parameters
            opt_params = []
            if len(parameters) > num_parameter_groups:
                net_params, opt_params = parameters[:10], parameters[10:]
            params_dict = zip(net.state_dict().keys(), net_params)
            state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
            set_sgd_optimizer_state(optimizer, opt_params)
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, optimizer, trainloader, epochs=1)
            return self.get_parameters(), len(trainloader), {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            print(float(loss), len(testloader), {"accuracy": float(accuracy)})
            return float(loss), len(testloader), {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())


if __name__ == "__main__":
    main()
