"""Define our models, and training and eval functions.

If your model is 100% off-the-shelf (e.g. directly from torchvision without requiring
modifications) you might be better off instantiating your  model directly from the Hydra
config. In this way, swapping your model for  another one can be done without changing
the python code at all
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Tuple

class LogisticRegression(nn.Module):
    """Simple logistic regression."""

    def __init__(self, num_classes, input_dim=28*28):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


from torch.optim import Optimizer

class ScaffoldOptimizer(Optimizer):
    """
    Implements optimizer step function as defined in the SCAFFOLD paper
    """

    def __init__(self, grads, step_size):
        super().__init__(grads, {"lr":step_size})
    
    def step(self, server_cv, client_cv):
        # y_i = y+i - eta * (g_i + c - c_i)
        for group in self.param_groups:
            for p, s, c in zip(group["params"], server_cv, client_cv):
                p.data.add_(p.grad.data + s - c, alpha=-group["lr"])

def train(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
) -> None:
    """Train the network on the training set.
    
    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    trainloader : DataLoader
        The training set dataloader object.
    device : torch.device
        The device on which to train the network.
    epochs : int
        The number of epochs to train the network.
    learning_rate : float
        The learning rate.
    server_cv : torch.Tensor
        The server's control variate.
    client_cv : torch.Tensor
        The client's control variate.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = ScaffoldOptimizer(net.parameters(), learning_rate)
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(
            net, trainloader, device, criterion, optimizer, server_cv, client_cv
        )

def _train_one_epoch(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: ScaffoldOptimizer,
    server_cv: torch.Tensor,
    client_cv: torch.Tensor,
) -> nn.Module:
    """Train the network on the training set for one epoch."""
    for data, target in trainloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step(server_cv, client_cv) # type: ignore
    return net

def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float]:
    """Evaluate the network on the test set.
    
    Parameters
    ----------
    net : nn.Module
        The neural network to evaluate.
    testloader : DataLoader
        The test set dataloader object.
    device : torch.device
        The device on which to evaluate the network.
    
    Returns
    -------
    Tuple[float, float]
        The loss and accuracy of the network on the test set.
    """

    criterion = nn.CrossEntropyLoss()
    net.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss = loss / total
    acc = correct / total
    return loss, acc