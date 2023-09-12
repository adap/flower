"""Abstract class for splitting a model into body and head."""
import copy
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


class ModelSplit(ABC, nn.Module):
    """Abstract class for splitting a model into body and head.

    Optionally, a fixed head can also be created.
    """

    def __init__(
        self,
        model: nn.Module,
        has_fixed_head: bool = False,
    ):
        """Initialize the attributes of the model split.

        Args:
            model: dict containing the vocab sizes of the input attributes.
            has_fixed_head: whether the model should contain a fixed_head.
        """
        super().__init__()

        self._body, self._head = self._get_model_parts(model)

        self._fixed_head = copy.deepcopy(self.head) if has_fixed_head else None
        self._use_fixed_head = False

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
        pass

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

    @property
    def fixed_head(self) -> Optional[nn.Module]:
        """Return model fixed_head."""
        return self._fixed_head

    @fixed_head.setter
    def fixed_head(self, state_dict: "OrderedDict[str, Tensor]") -> None:
        """Set model fixed_head.

        Args:
            state_dict: dictionary of the state to set the model fixed head to.
        """
        if self._fixed_head is None:
            # When the fixed_head was not initialized
            return
        self._fixed_head.load_state_dict(state_dict, strict=True)
        self.disable_fixed_head()

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

    def set_parameters(self, state_dict: Dict[str, Tensor]) -> None:
        """Set model parameters.

        Args:
            state_dict: dictionary of the state to set the model to.
        """
        ordered_state_dict = OrderedDict(self.state_dict().copy())
        # Update with the values of the state_dict
        ordered_state_dict.update(dict(state_dict.items()))
        self.load_state_dict(ordered_state_dict, strict=False)

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

    def disable_fixed_head(self) -> None:
        """Disable gradient tracking for the fixed head parameters."""
        if self._fixed_head is None:
            return
        for param in self._fixed_head.parameters():
            param.requires_grad = False

    def use_fixed_head(self, use_fixed_head: bool) -> None:
        """Set whether the fixed head should be used for forward.

        Args:
            use_fixed_head: boolean indicating whether to use the fixed head or not.
        """
        self._use_fixed_head = use_fixed_head

    def forward(self, inputs: Any) -> Any:
        """Forward inputs through the body and the head (or fixed head)."""
        x = self.body(inputs)
        x = F.relu(x)
        if self._use_fixed_head and self.fixed_head is not None:
            return self.fixed_head(x)
        return self.head(x)


class ModelManager(ABC):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_id: int,
        config: Dict[str, Any],
        model_split_class: Type[Any],  # ModelSplit
        has_fixed_head: bool = False,
    ):
        """Initialize the attributes of the model manager.

        Args:
            client_id: The id of the client.
            config: Dict containing the configurations to be used by the manager.
            model_split_class: Class to be used to split the model into body and head\
                (concrete implementation of ModelSplit).
            has_fixed_head: Whether a fixed head should be created.
        """
        super().__init__()

        self.client_id = client_id
        self.config = config
        self._model = model_split_class(self._create_model(), has_fixed_head)

    @abstractmethod
    def _create_model(self) -> nn.Module:
        """Return model to be splitted into head and body."""
        pass

    @abstractmethod
    def train(
        self,
        epochs: int = 1,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Args:
            epochs: number of training epochs.

        Returns
        -------
            Dict containing the train metrics.
        """
        pass

    @abstractmethod
    def test(self, test_id: int) -> Dict[str, float]:
        """Test the model maintained in self.model.

        Args:
            test_id: id of the test round.

        Returns
        -------
            Dict containing the test metrics.
        """
        pass

    @abstractmethod
    def train_dataset_size(self) -> int:
        """Return train data set size."""
        pass

    @abstractmethod
    def test_dataset_size(self) -> int:
        """Return test data set size."""
        pass

    @abstractmethod
    def total_dataset_size(self) -> int:
        """Return total data set size."""
        pass

    @property
    def model(self) -> nn.Module:
        """Return model."""
        return self._model
