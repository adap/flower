"""Abstract class for splitting a model into body and head."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

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


def get_device(
    use_cuda: bool = True, specified_device: Optional[int] = None
) -> torch.device:
    """Get the tensor device.

    Args:
        use_cuda: Flag indicates whether to use CUDA or not. Defaults to True.
        specified_device: Specified cuda device to use. Defaults to None.

    Raises
    ------
        ValueError: Specified device not in CUDA_VISIBLE_DEVICES.

    Returns
    -------
        The selected or fallbacked device.
    """
    device = torch.device("cpu")
    if use_cuda and torch.cuda.is_available():
        if specified_device is not None:
            cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
            if cuda_visible_devices is not None:
                devices = [int(d) for d in cuda_visible_devices.split(",")]
                if specified_device in devices:
                    device = torch.device(f"cuda:{specified_device}")
                else:
                    raise ValueError(
                        f"Specified device {specified_device}"
                        " not in CUDA_VISIBLE_DEVICES"
                    )
            else:
                print("CUDA_VISIBLE_DEVICES not exists, using torch.device('cuda').")
        else:
            device = torch.device("cuda")

    return device


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
    def body(self, state_dict: OrderedDict[str, Tensor]) -> None:
        """Set model body.

        Args:
            state_dict: dictionary of the state to set the model body to.
        """
        self._body.load_state_dict(state_dict, strict=True)

    @property
    def head(self) -> nn.Module:
        """Return model head."""
        return self._head

    @head.setter
    def head(self, state_dict: OrderedDict[str, Tensor]) -> None:
        """Set model head.

        Args:
            state_dict: dictionary of the state to set the model head to.
        """
        self._head.load_state_dict(state_dict, strict=True)

    def get_parameters(self) -> List[np.ndarray]:
        """Get model parameters.

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

    def set_parameters(self, state_dict: Dict[str, Tensor]) -> None:
        """Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        self.load_state_dict(state_dict, strict=False)

    def enable_head(self) -> None:
        """Enable gradient tracking for the head parameters."""
        for param in self._head.parameters():
            param.requires_grad = True

    def enable_body(self) -> None:
        """Enable gradient tracking for the body parameters."""
        for param in self._body.parameters():
            param.requires_grad = True

    def disable_head(self) -> None:
        """Disable gradient tracking for the head parameters."""
        for param in self._head.parameters():
            param.requires_grad = False

    def disable_body(self) -> None:
        """Disable gradient tracking for the body parameters."""
        for param in self._body.parameters():
            param.requires_grad = False

    def forward(self, inputs: Any) -> Any:
        """Forward inputs through the body and the head."""
        return self.head(self.body(inputs))


# pylint: disable=R0902, R0913, R0801
class ModelManager(ABC):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_id: int,
        config: DictConfig,
        trainloader: DataLoader,
        testloader: DataLoader,
        client_save_path: Optional[str],
        model_split_class: Any,  # ModelSplit
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            trainloader: Client train dataloader.
            testloader: Client test dataloader.
            client_save_path: Path to save the client model head state.
            model_split_class: Class to be used to split the model into body and head \
                (concrete implementation of ModelSplit).
        """
        super().__init__()
        self.config = config
        self.client_id = client_id
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = get_device(
            use_cuda=getattr(self.config, "use_cuda", True),
            specified_device=getattr(self.config, "specified_device", None),
        )
        self.client_save_path = client_save_path
        self.learning_rate = config.get("learning_rate", 0.01)
        self.momentum = config.get("momentum", 0.5)
        self._model: ModelSplit = model_split_class(self._create_model())

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return model to be splitted into head and body."""

    @property
    def model(self) -> ModelSplit:
        """Return model."""
        return self._model

    def train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Returns
        -------
            Dict containing the train metrics.
        """
        # Load client state (head) if client_save_path is not None and it is not empty
        if self.client_save_path is not None and os.path.isfile(self.client_save_path):
            self._model.head.load_state_dict(torch.load(self.client_save_path))

        num_local_epochs = DEFAULT_LOCAL_TRAIN_EPOCHS
        if hasattr(self.config, "num_local_epochs"):
            num_local_epochs = int(self.config.num_local_epochs)

        num_rep_epochs = DEFAULT_REPRESENTATION_EPOCHS
        if hasattr(self.config, "num_rep_epochs"):
            num_rep_epochs = int(self.config.num_rep_epochs)

        criterion = torch.nn.CrossEntropyLoss()
        weights = [v for k, v in self._model.named_parameters() if "weight" in k]
        biases = [v for k, v in self._model.named_parameters() if "bias" in k]
        optimizer = torch.optim.SGD(
            [
                {"params": weights, "weight_decay": 1e-4},
                {"params": biases, "weight_decay": 0.0},
            ],
            lr=self.learning_rate,
            momentum=self.momentum,
        )
        correct, total = 0, 0
        loss: torch.Tensor = 0.0

        self._model.train()
        for i in range(num_local_epochs + num_rep_epochs):
            if i < num_local_epochs:
                self._model.disable_body()
                self._model.enable_head()
            else:
                self._model.enable_body()
                self._model.disable_head()
            for batch in self.trainloader:
                images = batch["img"]
                labels = batch["label"]
                outputs = self._model(images.to(self.device))
                labels = labels.to(self.device)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        # Save client state (head)
        if self.client_save_path is not None:
            torch.save(self._model.head.state_dict(), self.client_save_path)

        return {"loss": loss.item(), "accuracy": correct / total}

    def test(self) -> Dict[str, float]:
        """Test the model maintained in self.model.

        Returns
        -------
            Dict containing the test metrics.
        """
        # Load client state (head)
        if self.client_save_path is not None and os.path.isfile(self.client_save_path):
            self._model.head.load_state_dict(torch.load(self.client_save_path))

        num_finetune_epochs = DEFAULT_FINETUNE_EPOCHS
        if hasattr(self.config, "num_finetune_epochs"):
            num_finetune_epochs = int(self.config.num_finetune_epochs)

        if num_finetune_epochs > 0 and self.config.get("enable_finetune", False):
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self.learning_rate)
            criterion = torch.nn.CrossEntropyLoss()
            self._model.train()
            for _ in range(num_finetune_epochs):
                for batch in self.trainloader:
                    images = batch["img"].to(self.device)
                    labels = batch["label"].to(self.device)
                    outputs = self._model(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0

        self._model.eval()
        with torch.no_grad():
            for batch in self.testloader:
                images = batch["img"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self._model(images)
                loss += criterion(outputs, labels).item()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

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
