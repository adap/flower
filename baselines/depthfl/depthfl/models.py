"""ResNet18 model architecutre, training, and testing functions for CIFAR100."""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class KLLoss(nn.Module):
    """KL divergence loss for self distillation."""

    def __init__(self):
        super().__init__()
        self.temperature = 1

    def forward(self, pred, label):
        """KL loss forward."""
        predict = F.log_softmax(pred / self.temperature, dim=1)
        target_data = F.softmax(label / self.temperature, dim=1)
        target_data = target_data + 10 ** (-7)
        with torch.no_grad():
            target = target_data.detach().clone()

        loss = (
            self.temperature
            * self.temperature
            * ((target * (target.log() - predict)).sum(1).sum() / target.size()[0])
        )
        return loss


def train(  # pylint: disable=too-many-arguments
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    config: dict,
    consistency_weight: float,
    prev_grads: dict,
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
    learning_rate : float
        The learning rate for the SGD optimizer.
    config : dict
        training configuration
    consistency_weight : float
        hyperparameter for self distillation
    prev_grads : dict
        control variate for feddyn
    """
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=1e-3)
    global_params = {
        k: val.detach().clone().flatten() for (k, val) in net.named_parameters()
    }

    for k, _ in net.named_parameters():
        prev_grads[k] = prev_grads[k].to(device)

    net.train()
    for _ in range(epochs):
        _train_one_epoch(
            net,
            global_params,
            trainloader,
            device,
            criterion,
            optimizer,
            config,
            consistency_weight,
            prev_grads,
        )

    # update prev_grads for FedDyn
    if config["feddyn"]:
        update_prev_grads(config, net, prev_grads, global_params)


def update_prev_grads(config, net, prev_grads, global_params):
    """Update prev_grads for FedDyn."""
    for k, param in net.named_parameters():
        curr_param = param.detach().clone().flatten()
        prev_grads[k] = prev_grads[k] - config["alpha"] * (
            curr_param - global_params[k]
        )
        prev_grads[k] = prev_grads[k].to(torch.device(torch.device("cpu")))


def _train_one_epoch(  # pylint: disable=too-many-locals, too-many-arguments
    net: nn.Module,
    global_params: dict,
    trainloader: DataLoader,
    device: torch.device,
    criterion: torch.nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
    config: dict,
    consistency_weight: float,
    prev_grads: dict,
):
    """Train for one epoch.

    Parameters
    ----------
    net : nn.Module
        The neural network to train.
    global_params : List[Parameter]
        The parameters of the global model (from the server).
    trainloader : DataLoader
        The DataLoader containing the data to train the network on.
    device : torch.device
        The device on which the model should be trained, either 'cpu' or 'cuda'.
    criterion : torch.nn.CrossEntropyLoss
        The loss function to use for training
    optimizer : torch.optim.Adam
        The optimizer to use for training
    config : dict
        training configuration
    consistency_weight : float
        hyperparameter for self distillation
    prev_grads : dict
        control variate for feddyn
    """
    criterion_kl = KLLoss().cuda()

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        loss = torch.zeros(1).to(device)
        optimizer.zero_grad()
        output_lst = net(images)

        for i, branch_output in enumerate(output_lst):
            # only trains last classifier in InclusiveFL
            if not config["extended"] and i != len(output_lst) - 1:
                continue

            loss += criterion(branch_output, labels)

            # self distillation term
            if config["kd"] and len(output_lst) > 1:
                for j, output in enumerate(output_lst):
                    if j == i:
                        continue

                    loss += (
                        consistency_weight
                        * criterion_kl(branch_output, output.detach())
                        / (len(output_lst) - 1)
                    )

        # Dynamic regularization in FedDyn
        if config["feddyn"]:
            for k, param in net.named_parameters():
                curr_param = param.flatten()

                lin_penalty = torch.dot(curr_param, prev_grads[k])
                loss -= lin_penalty

                quad_penalty = (
                    config["alpha"]
                    / 2.0
                    * torch.sum(torch.square(curr_param - global_params[k]))
                )
                loss += quad_penalty

        loss.backward()
        optimizer.step()


def test(  # pylint: disable=too-many-locals
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> Tuple[float, float, List[float]]:
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
    Tuple[float, float, List[float]]
        The loss and the accuracy of the global model
        and the list of accuracy for each classifier on the given data.
    """
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    correct_single = [0] * 4  # accuracy of each classifier within model
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            output_lst = net(images)

            # ensemble classfiers' output
            ensemble_output = torch.stack(output_lst, dim=2)
            ensemble_output = torch.sum(ensemble_output, dim=2) / len(output_lst)

            loss += criterion(ensemble_output, labels).item()
            _, predicted = torch.max(ensemble_output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i, single in enumerate(output_lst):
                _, predicted = torch.max(single, 1)
                correct_single[i] += (predicted == labels).sum().item()

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    accuracy_single = [correct / total for correct in correct_single]
    return loss, accuracy, accuracy_single


def test_sbn(  # pylint: disable=too-many-locals
    nets: List[nn.Module],
    trainloaders: List[DictConfig],
    testloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, List[float]]:
    """Evaluate the networks on the entire test set.

    Parameters
    ----------
    nets : List[nn.Module]
        The neural networks to test. Each neural network has different width
    trainloaders : List[DataLoader]
        The List of dataloaders containing the data to train the network on
    testloader : DataLoader
        The DataLoader containing the data to test the network on.
    device : torch.device
        The device on which the model should be tested, either 'cpu' or 'cuda'.

    Returns
    -------
    Tuple[float, float, List[float]]
        The loss and the accuracy of the global model
        and the list of accuracy for each classifier on the given data.
    """
    # static batch normalization
    for trainloader in trainloaders:
        with torch.no_grad():
            for model in nets:
                model.train()
                for _batch_idx, (images, labels) in enumerate(trainloader):
                    images, labels = images.to(device), labels.to(device)
                    output = model(images)

                model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    correct_single = [0] * 4

    # test each network of different width
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            output_lst = []

            for model in nets:
                output_lst.append(model(images)[0])

            output = output_lst[-1]

            loss += criterion(output, labels).item()
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i, single in enumerate(output_lst):
                _, predicted = torch.max(single, 1)
                correct_single[i] += (predicted == labels).sum().item()

    if len(testloader.dataset) == 0:
        raise ValueError("Testloader can't be 0, exiting...")
    loss /= len(testloader.dataset)
    accuracy = correct / total
    accuracy_single = [correct / total for correct in correct_single]
    return loss, accuracy, accuracy_single
