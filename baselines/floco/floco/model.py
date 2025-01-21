"""floco: A Flower Baseline."""

from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

class SimplexModel(torch.nn.Module):
    def __init__(self, endpoints, seed) -> None:
        super().__init__()
        self.endpoints = endpoints
        self.seed = seed
        self.base_model = Net()
        self.base_model.classifier = SimplexLinear(
            endpoints=self.endpoints,
            in_features=self.base_model.classifier.in_features,
            out_features=self.base_model.classifier.out_features,
            bias=True,
            seed=self.seed,
        )
        self.subregion_parameters = None
        self.training = False

    def forward(self, x):
        endpoints = self.endpoints
        if self.subregion_parameters is None:  # before projection
            if self.training:  # sample uniformly from simplex for training
                sample = np.random.exponential(scale=1.0, size=endpoints)
                self.base_model.classifier.alphas = sample / sample.sum()
            else:  # use simplex center for testing
                simplex_center = tuple([1 / endpoints for _ in range(endpoints)])
                self.base_model.classifier.alphas = simplex_center
        else:  # after projection
            if self.training:  # sample uniformly from subregion for training
                self.base_model.classifier.alphas = _sample_L1_ball(*self.subregion_parameters)
            else:  # use subregion center for testing
                self.base_model.classifier.alphas = self.subregion_parameters[0]
        return self.base_model(x)
    

class SimplexLinear(torch.nn.Linear):
    def __init__(self, endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoints = endpoints
        self.alphas = tuple([1 / endpoints for _ in range(endpoints)])
        self._weights = torch.nn.ParameterList(
            [_initialize_weight(self.weight, seed + i) for i in range(endpoints)]
        )

    @property
    def weight(self) -> torch.nn.Parameter:
        return sum(alpha * weight for alpha, weight in zip(self.alphas, self._weights))


def _sample_L1_ball(center, radius):
    u = np.random.uniform(-1, 1, len(center))
    u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
    return center + np.random.uniform(0, radius) * u


def _initialize_weight(init_weight: torch.Tensor, seed: int) -> torch.nn.Parameter:
    weight = torch.nn.Parameter(torch.zeros_like(init_weight))
    torch.manual_seed(seed)
    torch.nn.init.xavier_normal_(weight)
    return weight


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=0, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1)
        self.fc1 = nn.Linear(in_features=64 * 3 * 3, out_features=128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        """Do forward."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.relu(self.fc1(x))
        return self.classifier(x)


def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.2, momentum=0.5, weight_decay=5e-4)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            loss = criterion(net(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def get_weights(net):
    """Extract model parameters as numpy arrays from state_dict."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Apply parameters to an existing model."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
