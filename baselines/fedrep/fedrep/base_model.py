"""fedrep: A Flower Baseline."""

import collections
from abc import ABC, abstractmethod
from typing import Any, Dict, List, OrderedDict, Tuple, Union

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from flwr.common import Context, NDArrays, ParametersRecord, array_from_numpy

from .constants import (
    DEFAULT_FINETUNE_EPOCHS,
    DEFAULT_LOCAL_TRAIN_EPOCHS,
    DEFAULT_REPRESENTATION_EPOCHS,
    FEDREP_HEAD_STATE,
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

    def get_parameters(self) -> NDArrays:
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
        context: Context,
        trainloader: DataLoader,
        testloader: DataLoader,
        model_split_class: Any,  # ModelSplit
    ):
        """Initialize the attributes of the model manager.

        Args:
            context: The context of the current run.
            trainloader: Client train dataloader.
            testloader: Client test dataloader.
            model_split_class: Class to be used to split the model into body and head \
                (concrete implementation of ModelSplit).
        """
        super().__init__()
        self.context = context
        self.trainloader = trainloader
        self.testloader = testloader
        self.learning_rate = self.context.run_config.get("learning-rate", 0.01)
        self.momentum = self.context.run_config.get("momentum", 0.5)
        self._model: ModelSplit = model_split_class(self._create_model())

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return model to be split into head and body."""

    @property
    def model(self) -> ModelSplit:
        """Return model."""
        return self._model

    def _load_client_state(self) -> None:
        """Load client model head state from context state; used only by FedRep."""
        # First, check if the fedrep head state is set in the context state.
        if self.context.state.parameters_records.get(FEDREP_HEAD_STATE):
            state_dict = collections.OrderedDict(
                {
                    k: torch.from_numpy(v.numpy())
                    for k, v in self.context.state.parameters_records[
                        FEDREP_HEAD_STATE
                    ].items()
                }
            )
            # Second, check if the parameters records have values stored and load
            # the state; this check is useful for the first time the model is
            # tested and the head state might be empty.
            if state_dict:
                self._model.head.load_state_dict(state_dict)

    def _save_client_state(self) -> None:
        """Save client model head state inside context state; used only by FedRep."""
        # Check if the fedrep head state is set in the context state.
        if FEDREP_HEAD_STATE in self.context.state.parameters_records:
            head_state = self._model.head.state_dict()
            head_state_np = {k: v.detach().cpu().numpy() for k, v in head_state.items()}
            head_state_arr = collections.OrderedDict(
                {k: array_from_numpy(v) for k, v in head_state_np.items()}
            )
            head_state_prec = ParametersRecord(head_state_arr)
            self.context.state.parameters_records[FEDREP_HEAD_STATE] = head_state_prec

    def train(
        self, device: torch.device
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Returns
        -------
            Dict containing the train metrics.
        """
        # Load state.
        self._load_client_state()

        num_local_epochs = DEFAULT_LOCAL_TRAIN_EPOCHS
        if "num-local-epochs" in self.context.run_config:
            num_local_epochs = int(self.context.run_config["num-local-epochs"])

        num_rep_epochs = DEFAULT_REPRESENTATION_EPOCHS
        if self.context.run_config["num-rep-epochs"] in self.context.run_config:
            num_rep_epochs = int(self.context.run_config["num-rep-epochs"])

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

        self._model.to(device)
        self._model.train()
        for i in range(num_local_epochs + num_rep_epochs):
            if i < num_local_epochs:
                self._model.disable_body()
                self._model.enable_head()
            else:
                self._model.enable_body()
                self._model.disable_head()
            for batch in self.trainloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
                outputs = self._model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

        # Save state.
        self._save_client_state()

        return {"loss": loss.item(), "accuracy": correct / total}

    def test(self, device: torch.device) -> Dict[str, float]:
        """Test the model maintained in self.model.

        Returns
        -------
            Dict containing the test metrics.
        """
        # Load state.
        self._load_client_state()

        num_finetune_epochs = DEFAULT_FINETUNE_EPOCHS
        if "num-finetune-epochs" in self.context.run_config:
            num_finetune_epochs = int(self.context.run_config["num-finetune-epochs"])

        if num_finetune_epochs > 0 and self.context.run_config.get(
            "enable-finetune", False
        ):
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self.learning_rate)
            criterion = torch.nn.CrossEntropyLoss()
            self._model.train()
            for _ in range(num_finetune_epochs):
                for batch in self.trainloader:
                    images = batch["img"]
                    labels = batch["label"]
                    outputs = self._model(images)
                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0

        self._model.to(device)
        self._model.eval()
        with torch.no_grad():
            for batch in self.testloader:
                images = batch["img"].to(device)
                labels = batch["label"].to(device)
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
