"""Defines pytorch models."""

from abc import abstractmethod
from typing import List

import numpy as np
import torch
import torch.nn as nn

from dasha.resnet import resnet18


class ClassificationModel(nn.Module):
    """Abstract class for classification models."""

    def __init__(
        self, input_shape: List[int], *args, **kwargs
    ):  # pylint: disable=unused-argument
        super().__init__()

    @abstractmethod
    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of a model."""
        return


class LinearNet(nn.Module):
    """Simple linear model."""

    def __init__(
        self,
        num_input_features: int,
        num_output_features: int,
        init_with_zeros: bool = False,
    ) -> None:
        super().__init__()
        self._linear = nn.Linear(num_input_features, num_output_features)
        if init_with_zeros:
            self._linear.weight.data.fill_(0.0)
            self._linear.bias.data.fill_(0.0)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass of a model."""
        output_tensor = self._linear(input_tensor.view(input_tensor.shape[0], -1))
        return output_tensor


class NonConvexLoss(nn.Module):
    """A nonconvex loss from Tyurin A. et al., 2023 paper.

    [DASHA: Distributed Nonconvex Optimization with Communication Compression and
    Optimal Oracle Complexity] (
    https://openreview.net/forum?id=VA1YpcNr7ul)
    """

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pylint: disable=no-self-use
        """Forward pass of a model."""
        assert len(input_tensor.shape) == 1
        target = 2 * target - 1
        input_target = input_tensor * target
        probs = torch.sigmoid(input_target)
        loss = torch.square(1 - probs)
        loss = torch.mean(loss)
        return loss


class LinearNetWithNonConvexLoss(ClassificationModel):
    """Classification model with a nonconvex loss."""

    def __init__(self, input_shape: List[int], init_with_zeros: bool = False) -> None:
        super().__init__(input_shape)
        assert len(input_shape) == 1
        num_input_features = input_shape[0]
        self._net = LinearNet(
            num_input_features, num_output_features=1, init_with_zeros=init_with_zeros
        )
        self._loss = NonConvexLoss()

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of a model."""
        logits = self._net(input_tensor).flatten()
        return self._loss(logits, target)

    @torch.no_grad()
    def accuracy(self, input_tensor: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy."""
        logits = self._net(input_tensor).numpy().flatten()
        target = target.numpy()
        predictions = (logits >= 0.0).astype(np.int32)
        return np.sum(predictions == target) / len(target)


class ResNet18WithLogisticLoss(ClassificationModel):
    """Resnet18 with the logistic loss."""

    def __init__(self, input_shape: List[int], num_classes: int = 10) -> None:
        super().__init__(input_shape)
        self._net = resnet18(num_classes=num_classes)
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self, input_tensor: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Forward pass of a model."""
        logits = self._net(input_tensor)
        return self._loss(logits, target.long())

    @torch.no_grad()
    def accuracy(self, input_tensor: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate accuracy."""
        logits = self._net(input_tensor).cpu().numpy()
        target = target.cpu().numpy()
        predictions = np.argmax(logits, axis=-1)
        return np.sum(predictions == target) / len(target)
