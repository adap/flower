
import torch
import numpy as np

from typing import Dict, List, Union
from collections import OrderedDict
from FedPer.utils.new_utils import Algorithms
from FedPer.utils.base_client import BaseClient


class FedPerClient(BaseClient):
    """Implementation of Federated Learning with Personalization Layers (FedPer) Client."""

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local head parameters."""
        return [val.cpu().numpy() for _, val in self.model_manager.model.body.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the local body parameters to the received parameters.
        In the first train round the head parameters are also set to the global head parameters,
        to ensure every client head is initialized equally.

        Args:
            parameters: parameters to set the body to.
        """
        model_keys = [k for k in self.model_manager.model.state_dict().keys() if k.startswith("_body")]

        if self.train_id == 1:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend([k for k in self.model_manager.model.state_dict().keys() if k.startswith("_head")])

        params_dict = zip(model_keys, parameters)

        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        self.model_manager.model.set_parameters(state_dict)

    def perform_train(self, tag: str = None) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """
        Perform local training to the whole model.

        Args:
            tag: str of the form <Algorithm>_<model_train_part>.
                <Algorithm> - indicates the federated algorithm that is being performed\
                              (FedAvg, FedPer, FedRep, FedBABU or FedHybridAvgLGDual).
                              In the case of FedHybridAvgLGDual the tag also includes which part of the algorithm\
                                is being performed, either FedHybridAvgLGDual_FedAvg or FedHybridAvgLGDual_LG-FedAvg.
                <model_train_part> - indicates the part of the model that is being trained (full, body, head).
                This tag can be ignored if no difference in train behaviour is desired between federated algortihms.
        Returns:
            Dict with the train metrics.
        """
        return super().perform_train(tag=f"{Algorithms.FEDPER.value}_full" if tag is None else tag)
