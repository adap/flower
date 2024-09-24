"""Model definitions for FedPara."""

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from flwr.common import Scalar
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader


class LowRankNN(nn.Module):
    """Fedpara Low-rank weight systhesis for fully connected layer."""

    def __init__(self, input_, output, rank) -> None:
        super().__init__()

        self.X = nn.Parameter(
            torch.empty(size=(input_, rank)),
            requires_grad=True,
        )
        self.Y = nn.Parameter(torch.empty(size=(output, rank)), requires_grad=True)

        init.kaiming_normal_(self.X, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.Y, mode="fan_out", nonlinearity="relu")

    def forward(self):
        """Forward pass."""
        out = torch.einsum("yr,xr->yx", self.Y, self.X)
        return out


class Linear(nn.Module):
    """Low-rank fully connected layer module for personalized scheme."""

    def __init__(self, input_, output, ratio, bias=True) -> None:
        super().__init__()
        rank = self._calc_from_ratio(ratio, input_, output)
        self.w1 = LowRankNN(input_, output, rank)
        self.w2 = LowRankNN(input_, output, rank)
        # make the bias for each layer
        if bias:
            self.bias = nn.Parameter(torch.zeros(output))

    @staticmethod
    def _calc_from_ratio(ratio, input_, output):
        # Return the low-rank of sub-matrices given the compression ratio
        # minimum possible parameter
        r1 = int(np.ceil(np.sqrt(output)))
        r2 = int(np.ceil(np.sqrt(input_)))
        r = np.min((r1, r2))
        # maximum possible rank
        # To solve it we need to know the roots of quadratic equation: 2*r*(m+n)=m*n
        r3 = math.floor((output * input_) / (2 * (output + input_)))
        rank = math.ceil((1 - ratio) * r + ratio * r3)
        return rank

    def forward(self, x):
        """Forward pass."""
        # personalized
        w = self.w1() * self.w2() + self.w1()
        out = F.linear(x, w, self.bias)
        return out


class FC(nn.Module):
    """2NN Fully connected layer as in the paper: https://arxiv.org/abs/1602.05629."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        input_size=28**2,
        hidden_size=200,
        num_classes=10,
        ratio=0.5,
        param_type="standard",
    ):
        super().__init__()
        self.input_size = input_size
        if param_type == "standard":
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 256)
            self.out = nn.Linear(256, num_classes)

        elif param_type == "lowrank":
            self.fc1 = Linear(input_size, hidden_size, ratio)
            self.fc2 = Linear(hidden_size, 256, ratio)
            self.out = Linear(256, num_classes, ratio)

        else:
            raise ValueError("param_type must be either standard or lowrank")

    @property
    def model_size(self):
        """Return the total number of trainable parameters (in million paramaters) and.

        the size of the model in MB.
        """
        total_trainable_params = (
            sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        )
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return total_trainable_params, size_all_mb

    def forward(self, x):
        """Forward pass."""
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


class LowRank(nn.Module):
    """Fedpara Low-rank weight systhesis for Convolution layer."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        low_rank: int,
        kernel_size: int,
    ):
        super().__init__()
        self.T = nn.Parameter(
            torch.empty(size=(low_rank, low_rank, kernel_size, kernel_size)),
            requires_grad=True,
        )
        self.X = nn.Parameter(
            torch.empty(size=(low_rank, out_channels)), requires_grad=True
        )
        self.Y = nn.Parameter(
            torch.empty(size=(low_rank, in_channels)), requires_grad=True
        )
        init.kaiming_normal_(self.T, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.X, mode="fan_out", nonlinearity="relu")
        init.kaiming_normal_(self.Y, mode="fan_out", nonlinearity="relu")

    def forward(self):
        """Forward pass."""
        # torch.einsum simplify the tensor produce (matrix multiplication)
        return torch.einsum("xyzw,xo,yi->oizw", self.T, self.X, self.Y)


# pylint: disable=too-many-instance-attributes
class Conv2d(nn.Module):
    """Convolutional layer with low-rank weights."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        ratio: float = 0.1,
        add_nonlinear: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias
        self.ratio = ratio
        self.low_rank = self._calc_from_ratio()
        self.add_nonlinear = add_nonlinear
        self.W1 = LowRank(in_channels, out_channels, self.low_rank, kernel_size)
        self.W2 = LowRank(in_channels, out_channels, self.low_rank, kernel_size)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.tanh = nn.Tanh()

    def _calc_from_ratio(self):
        # Return the low-rank of sub-matrices given the compression ratio

        # minimum possible parameter
        r1 = int(np.ceil(np.sqrt(self.out_channels)))
        r2 = int(np.ceil(np.sqrt(self.in_channels)))
        r = np.min((r1, r2))

        # maximum possible rank,
        # To solve it we need to know the roots of quadratic equation: ax^2+bx+c=0
        # a = kernel**2
        # b = out channel+ in channel
        # c = - num_target_params/2
        # r3 is floored because we cannot take the ceil as it results a bigger number
        # of parameters than the original problem

        num_target_params = self.out_channels * self.in_channels * (self.kernel_size**2)
        a, b, c = (
            self.kernel_size**2,
            self.out_channels + self.in_channels,
            -num_target_params / 2,
        )
        discriminant = b**2 - 4 * a * c
        r3 = math.floor((-b + math.sqrt(discriminant)) / (2 * a))
        ratio = math.ceil((1 - self.ratio) * r + self.ratio * r3)
        return ratio

    def forward(self, x):
        """Forward pass."""
        # Hadamard product of two submatrices
        if self.add_nonlinear:
            W = self.tanh(self.W1()) * self.tanh(self.W2())
        else:
            W = self.W1() * self.W2()
        out = F.conv2d(
            input=x, weight=W, bias=self.bias, stride=self.stride, padding=self.padding
        )
        return out


# pylint: disable=too-many-instance-attributes
class VGG(nn.Module):
    """VGG16GN model."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        num_classes,
        num_groups=2,
        ratio=0.1,
        param_type="lowrank",
        add_nonlinear=False,
    ):
        super().__init__()
        self.param_type = param_type
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.ratio = ratio
        self.add_nonlinear = add_nonlinear
        self.features = self._make_layers(
            [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                "M",
            ]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights."""
        for name, module in self.features.named_children():
            module = getattr(self.features, name)
            if isinstance(module, nn.Conv2d):
                if self.param_type == "lowrank":
                    num_channels = module.in_channels
                    setattr(
                        self.features,
                        name,
                        Conv2d(
                            num_channels,
                            module.out_channels,
                            module.kernel_size[0],
                            module.stride[0],
                            module.padding[0],
                            module.bias is not None,
                            ratio=self.ratio,
                            add_nonlinear=self.add_nonlinear,
                            # send the activation function to the Conv2d class
                        ),
                    )
                elif self.param_type == "standard":
                    n = (
                        module.kernel_size[0]
                        * module.kernel_size[1]
                        * module.out_channels
                    )
                    module.weight.data.normal_(0, math.sqrt(2.0 / n))
                    module.bias.data.zero_()

    def _make_layers(self, cfg, group_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if group_norm:
                    layers += [
                        conv2d,
                        nn.GroupNorm(self.num_groups, v),
                        nn.ReLU(inplace=True),
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @property
    def model_size(self):
        """Return the total number of trainable parameters (in million paramaters) and.

        the size of the model in MB.
        """
        total_trainable_params = (
            sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        )
        param_size = 0
        for param in self.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024**2
        return total_trainable_params, size_all_mb

    def forward(self, x):
        """Forward pass."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Create an instance of the VGG16GN model with Group Normalization,
# custom Conv2d, and modified classifier
def test(
    net: nn.Module, test_loader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    test_loader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    if len(test_loader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(test_loader.dataset)
    accuracy = correct / total
    return loss, accuracy


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    hyperparams: Dict[str, Scalar],
    epoch: int,
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
    hyperparams : Dict[str, Scalar]
        The hyperparameters to use for training.
    """
    lr = float(hyperparams["eta_l"]) * float(hyperparams["learning_decay"]) ** (
        epoch - 1
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=lr,
        momentum=0,
        weight_decay=0,
    )
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(
            net=net,
            trainloader=trainloader,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
        )


def _train_one_epoch(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion,
    optimizer,
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
    criterion :
        The loss function to use for training
    optimizer :
        The optimizer to use for training
    hyperparams : Dict[str, Scalar]
        The hyperparameters to use for training.

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        net.zero_grad()
        log_probs = net(images)
        loss = criterion(log_probs, labels)
        loss.backward()
        optimizer.step()
    return net


if __name__ == "__main__":
    model = VGG(num_classes=10, num_groups=2, param_type="standard", ratio=0.4)
    # Print the modified VGG16GN model architecture
    print(model.model_size)
