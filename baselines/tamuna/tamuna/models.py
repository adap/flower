"""CNN model architecutre, training, and testing functions for MNIST."""


from typing import List, Tuple
from collections import OrderedDict

import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader

from utils import apply_nn_compression


class Net(nn.Module):
    """Convolutional Neural Network architecture.

    As described in McMahan 2017 paper :

    [Communication-Efficient Learning of Deep Networks from
    Decentralized Data] (https://arxiv.org/pdf/1602.05629.pdf)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = torch.flatten(output_tensor, 1)
        output_tensor = F.relu(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor


def __model_zeroed_out(net):
    control_variate = copy.deepcopy(net)
    state_dict = OrderedDict(
        {k: torch.zeros_like(v) for k, v in net.state_dict().items()}
    )
    control_variate.load_state_dict(state_dict, strict=True)
    return control_variate


def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    eta: float,
    server_net: nn.Module,
    control_variate: nn.Module,
    old_compression_mask: torch.tensor,
    old_compressed_net: nn.Module,
) -> (nn.Module, nn.Module):
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
    learning_rate : float
        The learning rate for the SGD optimizer.
    proximal_mu : float
        Parameter for the weight of the proximal term.
    """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()

    if control_variate is not None:
        with torch.no_grad():
            control_variate.to(device)

            old_compressed_modules = []
            for module in list(old_compressed_net.modules())[1:]:
                if len(list(module.parameters())) != 0:
                    old_compressed_modules.append(module)

            server_net = apply_nn_compression(server_net, old_compression_mask)

            server_modules = []
            for server_module in list(server_net.modules())[1:]:
                if len(list(server_module.parameters())) != 0:
                    server_modules.append(server_module)

            control_variate_modules = []
            for control_variate_module in list(control_variate.modules())[1:]:
                if len(list(control_variate_module.parameters())) != 0:
                    control_variate_modules.append(control_variate_module)

            for i, module in enumerate(control_variate_modules):
                module.weight.copy_(
                    module.weight.data
                    + (eta / lr)
                    * (
                        server_modules[i].weight.data
                        - old_compressed_modules[i].weight.data
                    )
                )
                module.bias.copy_(
                    module.bias.data
                    + (eta / lr)
                    * (
                        server_modules[i].bias.data
                        - old_compressed_modules[i].bias.data
                    )
                )
    else:
        control_variate = __model_zeroed_out(net)

    for _ in range(epochs):
        net = _train_one_epoch(
            net,
            trainloader,
            device,
            criterion,
            lr,
            control_variate,
        )

    return net, control_variate


def _train_one_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    lr,
    control_variate,
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
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training

    Returns
    -------
    nn.Module
        The model that has been trained for one epoch.
    """

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        net.zero_grad(set_to_none=False)

        loss = criterion(net(images), labels)
        loss.backward()

        with torch.no_grad():
            control_variate.to(device)

            modules = []
            for module in list(net.modules())[1:]:
                if len(list(module.parameters())) != 0:
                    modules.append(module)

            control_variate_modules = []
            for control_variate_module in list(control_variate.modules())[1:]:
                if len(list(control_variate_module.parameters())) != 0:
                    control_variate_modules.append(control_variate_module)

            for i, module in enumerate(modules):
                module.weight.copy_(
                    module.weight.data
                    - lr
                    * (module.weight.grad.data - control_variate_modules[i].weight.data)
                )
                module.bias.copy_(
                    module.bias.data
                    - lr
                    * (module.bias.grad.data - control_variate_modules[i].bias.data)
                )
    return net


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the entire test set.

    Parameters
    ----------
    net : nn.Module
        The neural network to test.
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float]
        The loss and the accuracy of the input model on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
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
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy
