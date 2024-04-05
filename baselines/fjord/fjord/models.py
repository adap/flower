"""ResNet model for Fjord."""

from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from .od.models.utils import (
    SequentialWithSampler,
    create_bn_layer,
    create_conv_layer,
    create_linear_layer,
)
from .od.samplers import BaseSampler, ODSampler


class BasicBlock(nn.Module):
    """Basic Block for resnet."""

    expansion = 1

    def __init__(
        self, od, p_s, in_planes, planes, stride=1
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.od = od
        self.conv1 = create_conv_layer(
            od,
            True,
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = create_bn_layer(od=od, p_s=p_s, num_features=planes)
        self.conv2 = create_conv_layer(
            od, True, planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = create_bn_layer(od=od, p_s=p_s, num_features=planes)

        self.shortcut = SequentialWithSampler()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = SequentialWithSampler(
                create_conv_layer(
                    od,
                    True,
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                create_bn_layer(od=od, p_s=p_s, num_features=self.expansion * planes),
            )

    def forward(self, x, sampler):
        """Forward method for basic block.

        Args:
        :param x: input
        :param sampler: sampler
        :return: Output of forward pass
        """
        if sampler is None:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += self.shortcut(x)
            out = F.relu(out)
        else:
            out = F.relu(self.bn1(self.conv1(x, p=sampler())))
            out = self.bn2(self.conv2(out, p=sampler()))
            shortcut = self.shortcut(x, sampler=sampler)
            assert (
                shortcut.shape == out.shape
            ), f"Shortcut shape: {shortcut.shape} out.shape: {out.shape}"
            out += shortcut
            # out += self.shortcut(x, sampler=sampler)
            out = F.relu(out)
        return out


# Adapted from:
#   https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
class ResNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """ResNet in PyTorch.

    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition. arXiv:1512.03385
    """

    def __init__(
        self, od, p_s, block, num_blocks, num_classes=10
    ):  # pylint: disable=too-many-arguments
        super().__init__()
        self.od = od
        self.in_planes = 64

        self.conv1 = create_conv_layer(
            od, True, 3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = create_bn_layer(od=od, p_s=p_s, num_features=64)
        self.layer1 = self._make_layer(od, p_s, block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(od, p_s, block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(od, p_s, block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(od, p_s, block, 512, num_blocks[3], stride=2)
        self.linear = create_linear_layer(od, False, 512 * block.expansion, num_classes)

    def _make_layer(
        self, od, p_s, block, planes, num_blocks, stride
    ):  # pylint: disable=too-many-arguments
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for strd in strides:
            layers.append(block(od, p_s, self.in_planes, planes, strd))
            self.in_planes = planes * block.expansion
        return SequentialWithSampler(*layers)

    def forward(self, x, sampler=None):
        """Forward method for ResNet.

        Args:
        :param x: input
        :param sampler: sampler
        :return: Output of forward pass
        """
        if self.od:
            if sampler is None:
                sampler = BaseSampler(self)
            out = F.relu(self.bn1(self.conv1(x, p=sampler())))
            out = self.layer1(out, sampler=sampler)
            out = self.layer2(out, sampler=sampler)
            out = self.layer3(out, sampler=sampler)
            out = self.layer4(out, sampler=sampler)
            out = F.avg_pool2d(out, 4)  # pylint: disable=not-callable
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)  # pylint: disable=not-callable
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def ResNet18(od=False, p_s=(1.0,)):
    """Construct a ResNet-18 model.

    Args:
    :param od: whether to create OD (Ordered Dropout) layer
    :param p_s: list of p-values
    """
    return ResNet(od, p_s, BasicBlock, [2, 2, 2, 2])


def get_net(
    model_name: str,
    p_s: List[float],
    device: torch.device,
) -> torch.nn.Module:
    """Initialise model.

    :param model_name: name of the model
    :param p_s: list of p-values
    :param device: device to be used
    :return: initialised model
    """
    if model_name == "resnet18":
        net = ResNet18(od=True, p_s=p_s).to(device)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    return net


def train(  # pylint: disable=too-many-locals, too-many-arguments
    net: Module,
    trainloader: DataLoader,
    know_distill: bool,
    max_p: float,
    current_round: int,
    total_rounds: int,
    p_s: List[float],
    epochs: int,
    train_config: SimpleNamespace,
) -> float:
    """Train the model on the training set.

    :param net: The model to train.
    :param trainloader: The training set.
    :param know_distill: Whether the model being trained uses knowledge distillation.
    :param max_p: The maximum p value.
    :param current_round: The current round of training.
    :param total_rounds: The total number of rounds of training.
    :param p_s: The p values to use for training.
    :param epochs: The number of epochs to train for.
    :param train_config: The training configuration.
    :return: The loss on the training set.
    """
    device = next(net.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    if train_config.optimiser == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=train_config.lr,
            momentum=train_config.momentum,
            nesterov=train_config.nesterov,
            weight_decay=train_config.weight_decay,
        )
    else:
        raise ValueError(f"Optimiser {train_config.optimiser} not supported")
    lr_scheduler = get_lr_scheduler(
        optimizer, total_rounds, method=train_config.lr_scheduler
    )
    for _ in range(current_round):
        lr_scheduler.step()

    sampler = ODSampler(
        p_s=p_s,
        max_p=max_p,
        model=net,
    )
    max_sampler = ODSampler(
        p_s=[max_p],
        max_p=max_p,
        model=net,
    )

    loss = 0.0
    samples = 0
    for _ in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            target = labels.to(device)
            images = images.to(device)
            batch_size = images.shape[0]
            if know_distill:
                full_output = net(images.to(device), sampler=max_sampler)
                full_loss = criterion(full_output, target)
                full_loss.backward()
                target = full_output.detach().softmax(dim=1)
            partial_loss = criterion(net(images, sampler=sampler), target)
            partial_loss.backward()
            optimizer.step()
            loss += partial_loss.item() * batch_size
            samples += batch_size

    return loss / samples


def test(
    net: Module, testloader: DataLoader, p_s: List[float]
) -> Tuple[List[float], List[float]]:
    """Validate the model on the test set.

    :param net: The model to validate.
    :param testloader: The test set.
    :param p_s: The p values to use for validation.
    :return: The loss and accuracy on the test set.
    """
    device = next(net.parameters()).device
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    accuracies = []
    net.eval()

    for p in p_s:
        correct, loss = 0, 0.0
        p_sampler = ODSampler(
            p_s=[p],
            max_p=p,
            model=net,
        )

        with torch.no_grad():
            for images, labels in tqdm(testloader):
                outputs = net(images.to(device), sampler=p_sampler)
                labels = labels.to(device)
                loss += criterion(outputs, labels).item() * images.shape[0]
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        accuracy = correct / len(testloader.dataset)
        losses.append(loss / len(testloader.dataset))
        accuracies.append(accuracy)

    return losses, accuracies


def get_lr_scheduler(
    optimiser: Optimizer,
    total_epochs: int,
    method: Optional[str] = "static",
) -> torch.optim.lr_scheduler.LRScheduler:
    """Get the learning rate scheduler.

    :param optimiser: The optimiser for which to get the scheduler.
    :param total_epochs: The total number of epochs.
    :param method: The method to use for the scheduler. Supports static and cifar10.
    :return: The learning rate scheduler.
    """
    if method == "static":
        return MultiStepLR(optimiser, [total_epochs + 1])
    if method == "cifar10":
        return MultiStepLR(
            optimiser, [int(0.5 * total_epochs), int(0.75 * total_epochs)], gamma=0.1
        )
    raise ValueError(f"{method} scheduler not currently supported.")
