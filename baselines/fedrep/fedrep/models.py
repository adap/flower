"""Abstract class for splitting a model into body and head."""

import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import DataLoader

from fedrep.constants import (
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_LOCAL_TRAIN_EPOCHS,
    DEFAULT_REPRESENTATION_EPOCHS,
)


class ModelSplit(ABC, nn.Module):
    """Abstract class for splitting a model into body and head."""

    def __init__(self, model: nn.Module):
        """Initialize the attributes of the model split.

        Args:
            model: dict containing the vocab sizes of the input attributes.
        """
        super().__init__()

        self._body, self._head = self._get_model_parts(model)

    @abstractmethod
    def _get_model_parts(self, model: nn.Module) -> Tuple[nn.Module, nn.Module]:
        """Return the body and head of the model.

        Args:
            model: model to be split into head and body

        Returns
        -------
            Tuple where the first element is the body of the model
            and the second is the head.
        """

    @property
    def body(self) -> nn.Module:
        """Return model body."""
        return self._body

    @body.setter
    def body(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model body.

        Args:
            state_dict: dictionary of the state to set the model body to.
        """
        self.body.load_state_dict(state_dict, strict=True)

    @property
    def head(self) -> nn.Module:
        """Return model head."""
        return self._head

    @head.setter
    def head(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model head.

        Args:
            state_dict: dictionary of the state to set the model head to.
        """
        self.head.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters (without fixed head).

        Returns
        -------
            Body and head parameters
        """
        return [
            val.cpu().numpy()
            for val in [
                *self.body.state_dict().values(),
                *self.head.state_dict().values(),
            ]
        ]

    def set_parameters(self, state_dict: OrderedDict[str, Tensor]) -> None:
        """Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        self.load_state_dict(state_dict, strict=False)

    def enable_head(self) -> None:
        """Enable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = True

    def enable_body(self) -> None:
        """Enable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = True

    def disable_head(self) -> None:
        """Disable gradient tracking for the head parameters."""
        for param in self.head.parameters():
            param.requires_grad = False

    def disable_body(self) -> None:
        """Disable gradient tracking for the body parameters."""
        for param in self.body.parameters():
            param.requires_grad = False

    def forward(self, x):
        """Forward pass of the model."""
        x = self.body(x)
        x = self.head(x)
        # Basically we don't need to do log softmax explicitly,
        # as it is done by nn.CrossEntropyLoss()
        # So we actually should just return the x directly.
        # return x

        # However the official implementation did that,
        # so I leave this as it is.
        return torch.nn.functional.log_softmax(x, dim=1)


class ModelManager(ABC):
    """Manager for models with Body/Head split."""

    # pylint: disable=R0902
    default_local_train_epochs = DEFAULT_LOCAL_TRAIN_EPOCHS
    default_finetune_epochs = DEFAULT_FINETUNE_EPOCHS
    default_representation_epochs = DEFAULT_REPRESENTATION_EPOCHS

    # pylint: disable=R0902,R0913
    def __init__(
        self,
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        client_save_path: Optional[str],
        learning_rate: float,
        model_split_class: Type[Any],  # ModelSplit
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            model_split_class: Class to be used to split the model into body and head\
                (concrete implementation of ModelSplit).
        """
        super().__init__()

        self.client_id = client_id
        self.config = config
        self.trainloader, self.testloader = trainloader, testloader
        self.device = self.config.server_device
        self.client_save_path = client_save_path
        self.learning_rate = learning_rate
        self._model: ModelSplit = model_split_class(self._create_model())

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return model to be splitted into head and body."""

    def train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Method adapted from
        https://github.com/rahulv0205/fedrep_experiments/blob/main/models/Nets.py.

        Returns
        -------
            Dict containing the train metrics.
        """
        # Load client state (head) if client_save_path is not None and it is not empty
        algorithm = self.config.algorithm.lower()
        local_epochs = self.default_local_train_epochs
        rep_epochs = self.default_representation_epochs if algorithm == "fedrep" else 0
        if hasattr(self.config, "num_local_epochs"):
            local_epochs = int(self.config.num_local_epochs)
        if hasattr(self.config, "rep_epochs"):
            rep_epochs = int(self.config.rep_epochs)

        if algorithm == "fedrep":
            if self.client_save_path is not None and os.path.isfile(
                self.client_save_path
            ):
                self.model.head.load_state_dict(torch.load(self.client_save_path))
            else:
                print("No client state found, training from scratch.")
        criterion = torch.nn.CrossEntropyLoss()
        weights = [v for k, v in self.model.named_parameters() if "weight" in k]
        biases = [v for k, v in self.model.named_parameters() if "bias" in k]
        optimizer = torch.optim.SGD(
            [{"params": weights, "weight_decay": 0.0001}, {"params": biases}],
            lr=self.learning_rate,
            momentum=0,
        )
        correct, total = 0, 0
        loss: torch.Tensor = 0.0
        self.model.train()
        for i in range(local_epochs + rep_epochs):
            if self.config.algorithm.lower() == "fedrep":
                if i < local_epochs:
                    self.model.enable_head()
                    self.model.disable_body()
                else:
                    self.model.disable_head()
                    self.model.enable_body()
            for images, labels in self.trainloader:
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += labels.size(0)
                correct += (torch.argmax(outputs.data, 1) == labels).sum().item()

        # Save client state (head)
        if algorithm == "fedrep" and self.client_save_path is not None:
            torch.save(self.model.head.state_dict(), self.client_save_path)

        return {"loss": loss.item(), "accuracy": correct / total}

    def test(self) -> Dict[str, float]:
        """Test the model maintained in self.model.

        Returns
        -------
            Dict containing the test metrics.
        """
        # Load client state (head)
        algorithm = self.config.algorithm.lower()
        if (
            algorithm == "fedrep"
            and self.client_save_path is not None
            and os.path.isfile(self.client_save_path)
        ):
            self.model.head.load_state_dict(torch.load(self.client_save_path))

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.testloader:
                outputs = self.model(images.to(self.device))
                labels = labels.to(self.device)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.argmax(outputs.data, 1) == labels).sum().item()
        print("Test Accuracy: {:.4f}".format(correct / total))

        if algorithm == "fedrep" and self.client_save_path is not None:
            torch.save(self.model.head.state_dict(), self.client_save_path)

        return {
            "loss": loss / len(self.testloader.dataset),
            "accuracy": correct / total,
        }

    def train_dataset_size(self) -> int:
        """Return train data set size."""
        return len(self.trainloader.dataset)

    def test_dataset_size(self) -> int:
        """Return test data set size."""
        return len(self.testloader.dataset)

    def total_dataset_size(self) -> int:
        """Return total data set size."""
        return len(self.trainloader.dataset) + len(self.testloader.dataset)

    @property
    def model(self) -> nn.Module:
        """Return model."""
        return self._model
