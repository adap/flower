import copy
import flwr as fl
import torch 
import numpy as np

from typing import Dict, Any, Type, List, Union, Tuple
from collections import defaultdict, OrderedDict
from flwr.common import Scalar
from torch.utils.data import DataLoader
from FedPer.new_models import CNNModelManager
from FedPer.utils.constants import DEFAULT_FT_EP, DEFAULT_TRAIN_EP
# from FedPer.utils.model_manager import ModelManager

class BaseClient(fl.client.NumPyClient):
    """Implementation of Federated Averaging (FedAvg) Client."""

    def __init__(
        self,
        trainloader: DataLoader,
        testloader: DataLoader, 
        config: Dict[str, Any],
        client_id: int,
        model_manager_class: Type[CNNModelManager],
        has_fixed_head: bool = False,
    ):
        """
        Initialize client attributes.

        Args:
            config: dictionary containing the client configurations.
            client_id: id of the client.
            model_manager_class: class to be used as the model manager.
            has_fixed_head: whether a fixed head should be used or not.
        """
        super().__init__()

        self.train_id = 1
        self.test_id = 1
        self.config = config
        self.hist: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.model_manager = model_manager_class(
            client_id=client_id,
            config=config,
            has_fixed_head=has_fixed_head, 
            trainloader=trainloader,
            testloader=testloader
        )

    def get_parameters(self) -> List[np.ndarray]:
        """Return the current local model parameters."""
        return self.model_manager.model.get_parameters()

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set the local model parameters to the received parameters.

        Args:
            parameters: parameters to set the model to.
        """
        model_keys = [k for k in self.model_manager.model.state_dict().keys()
                      if k.startswith("_body") or k.startswith("_head")]
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
        epochs = self.config.get("epochs", {"full": DEFAULT_TRAIN_EP})
        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()
        return self.model_manager.train(
            train_id=self.train_id,
            epochs=epochs.get("full", DEFAULT_TRAIN_EP),
            tag="FedAvg_full" if tag is None else tag
        )

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Union[
        Tuple[List[np.ndarray], int, Dict[str, Scalar]],
        Tuple[List[np.ndarray], int]
    ]:
        """
        Train the provided parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns:
            Tuple containing the locally updated model parameters, \
                the number of examples used for training and \
                the training metrics.
        """
        self.set_parameters(parameters)

        train_results = self.perform_train()

        # Update train history
        self.hist[str(self.train_id)] = {**self.hist[str(self.train_id)], "trn": train_results}

        self.train_id += 1

        return self.get_parameters(), self.model_manager.train_dataset_size(), {}

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict[str, Scalar]
    ) -> Union[
        Tuple[float, int, Dict[str, Scalar]],
        Tuple[int, float, float],
        Tuple[int, float, float, Dict[str, Scalar]],
    ]:
        """
        Evaluate the provided global parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns:
        Tuple containing the test loss, \
                the number of examples used for evaluation and \
                the evaluation metrics.
        """
        self.set_parameters(parameters)

        if self.config.get("fine-tuning", False):
            # Save the model parameters before fine-tuning
            model_state_dict = copy.deepcopy(self.model_manager.model.state_dict())

            # Fine-tune the model before testing
            epochs = self.config.get("epochs", {"fine-tuning": DEFAULT_FT_EP})
            ft_trn_results = self.model_manager.train(
                train_id=self.test_id,
                epochs=epochs.get("fine-tuning", DEFAULT_FT_EP),
                fine_tuning=True,
                tag=f"{self.config.get('algorithm', 'FedAvg')}_full"
            )

        # Test the model
        tst_results = self.model_manager.test(
            test_id=self.test_id
        )

        # Update test history
        self.hist[str(self.test_id)] = {**self.hist[str(self.test_id)], "tst": tst_results}

        if self.config.get("fine-tuning", False):
            # Set the model parameters as they were before fine-tuning
            self.model_manager.model.set_parameters(model_state_dict)

            # Update the history with the ft_trn results
            self.hist[str(self.test_id)]["trn"] = {**(self.hist[str(self.test_id)].get("trn", {})), **ft_trn_results}
        self.test_id += 1

        return tst_results.get('loss', 0.0),\
            self.model_manager.test_dataset_size(),\
            {k: v for k, v in tst_results.items() if not isinstance(v, (dict, list))}
