"""floco: A Flower Baseline."""

from collections import OrderedDict
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from torch import nn


class SimplexModel(torch.nn.Module):
    """A wrapper that creates PyTroch model with a simplex classifier."""

    def __init__(self, endpoints, seed) -> None:
        super().__init__()
        self.endpoints = endpoints
        self.seed = seed
        self.base_model = Net(seed=self.seed)
        self.base_model.classifier = SimplexLinear(
            endpoints=self.endpoints,
            in_features=self.base_model.classifier.in_features,
            out_features=self.base_model.classifier.out_features,
            bias=True,
            seed=self.seed,
        )
        self.subregion_parameters: Optional[Tuple[ndarray, float]] = None
        self.training: bool = True

    def forward(self, x):
        """Compute forward pass through the model."""
        if self.subregion_parameters is None:  # before projection
            if self.training:  # sample uniformly from simplex for training
                sample = np.random.exponential(scale=1.0, size=self.endpoints)
                self.base_model.classifier.alphas = tuple(sample / sample.sum())
            else:  # use simplex center for testing
                simplex_center = [1 / self.endpoints for _ in range(self.endpoints)]
                self.base_model.classifier.alphas = simplex_center
        else:  # after projection
            if self.training:  # sample uniformly from subregion for training'
                self.base_model.classifier.alphas = tuple(
                    _sample_l1_ball(*self.subregion_parameters)
                )
            else:  # use subregion center for testing
                self.base_model.classifier.alphas = tuple(self.subregion_parameters[0])
        return self.base_model(x)


class SimplexLinear(torch.nn.Linear):
    """A fully-connected simplex classifier layer."""

    def __init__(self, endpoints: int, seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.endpoints = endpoints
        self.alphas = [1 / endpoints for _ in range(endpoints)]
        self._weights = torch.nn.ParameterList(
            [_initialize_weight(self.weight, seed + i) for i in range(endpoints)]
        )

    @property
    def weight(self) -> torch.nn.Parameter:
        """Return the weighted sum of the parameter weights."""
        return sum(alpha * weight for alpha, weight in zip(self.alphas, self._weights))


def _sample_l1_ball(center, radius):
    """Sample uniformly from L1 ball."""
    u = np.random.uniform(-1, 1, len(center))
    u = np.sign(u) * (np.abs(u) / np.sum(np.abs(u)))
    return center + np.random.uniform(0, radius) * u


def _initialize_weight(init_weight: torch.Tensor, seed: int) -> torch.nn.Parameter:
    """Initialize a weight tensor with Xavier normal initialization."""
    weight = torch.nn.Parameter(torch.zeros_like(init_weight))
    torch.manual_seed(seed)
    torch.nn.init.xavier_normal_(weight)
    return weight


class Net(nn.Module):
    """A simple CNN model for CIFAR-10."""

    def __init__(self, seed):
        super().__init__()
        bias = True
        self.conv1 = StandardConv(
            in_channels=3,
            out_channels=16,
            kernel_size=5,
            padding=0,
            stride=1,
            bias=bias,
        ).seed(seed)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = StandardConv(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            padding=1,
            stride=1,
            bias=bias,
        ).seed(seed)
        self.conv3 = StandardConv(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=1,
            bias=bias,
        ).seed(seed)
        self.fc1 = StandardLinear(
            in_features=64 * 3 * 3, out_features=128, bias=bias
        ).seed(seed)
        self.classifier = StandardLinear(
            in_features=128, out_features=10, bias=bias
        ).seed(seed)

    def forward(self, x):
        """Compute forward pass through the model."""
        x = self.pool(F.leaky_relu(self.conv1(x)))
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = self.pool(F.leaky_relu(self.conv3(x)))
        x = x.view(-1, 64 * 3 * 3)
        x = F.leaky_relu(self.fc1(x))
        return self.classifier(x)


def train(
    net, trainloader, epochs, device, reg_params=None, lamda=0.0
):  # pylint: disable=R0917
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = torch.optim.SGD(
        net.parameters(), lr=0.02, momentum=0.5, weight_decay=5e-4
    )
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            loss = criterion(net(images.to(device)), labels.to(device))
            optimizer.zero_grad()
            loss.backward()
            if reg_params is not None:
                _regularize_pers_model(net, reg_params, lamda)
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
    state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def _regularize_pers_model(model, reg_model_params, lamda):
    """Regularize the personal model with the global model."""
    for pers_param, global_param in zip(model.parameters(), reg_model_params):
        if pers_param.requires_grad and pers_param.grad is not None:
            pers_param.grad.data += lamda * (pers_param.data - global_param.data)


def seed_weights(weights: list, seed: int) -> None:
    """Seed the weights of a list of nn.Parameter objects."""
    for i, weight in enumerate(weights):
        torch.manual_seed(seed + i)
        torch.nn.init.xavier_normal_(weight)


class StandardConv(nn.Conv2d):
    """A standard convolutional layer."""

    def __int__(self, *args, **kwargs):
        """Initialize standard convolutional layer."""
        super().__init__(*args, **kwargs)

    def seed(self, s):
        """Seed the weights of the convolutional layer."""
        seed_weights([self.weight], s)
        return self


class StandardLinear(nn.Linear):
    """A standard fully-connected layer."""

    def __int__(self, *args, **kwargs):
        """Initialize standard linear layer."""
        super().__init__(*args, **kwargs)

    def seed(self, s):
        """Seed the weights of the linear layer."""
        seed_weights([self.weight], s)
        return self
