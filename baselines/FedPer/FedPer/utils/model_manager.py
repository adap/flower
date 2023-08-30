from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

import torch.nn as nn

from FedPer.utils.model_split import ModelSplit


class ModelManager(ABC):
    """Manager for models with Body/Head split."""

    def __init__(
        self,
        client_id: int,
        config: Dict[str, Any],
        model_split_class: Type[ModelSplit],
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
        train_id: int,
        epochs: int = 1,
        tag: Optional[str] = None,
        fine_tuning: bool = False,
    ) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Train the model maintained in self.model.

        Args:
            train_id: id of the train round.
            epochs: number of training epochs.
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                              (FedAvg, FedPer, FedRep, FedBABU or FedHybridAvgLGDual).
                              In the case of FedHybridAvgLGDual the tag also includes which part of the algorithm\
                                is being performed, either FedHybridAvgLGDual_FedAvg or FedHybridAvgLGDual_LG-FedAvg.
                <model_train_part> - indicates the part of the model that is being trained (full, body, head).
                This tag can be ignored if no difference in train behaviour is desired between federated algortihms.
            fine_tuning: whether the training performed is for model fine-tuning or not.

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
