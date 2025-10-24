"""fedrep: A Flower Baseline."""

from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import torch

from flwr.client import ClientApp, NumPyClient
from flwr.client.client import Client
from flwr.common import Context, NDArrays, ParametersRecord, Scalar

from .constants import FEDREP_HEAD_STATE, Algorithm
from .dataset import load_data
from .models import CNNCifar10ModelManager, CNNCifar100ModelManager
from .utils import get_model_manager_class


class BaseClient(NumPyClient):
    """Implementation of Federated Averaging (FedAvg) Client."""

    # pylint: disable=R0913
    def __init__(
        self, model_manager: Union[CNNCifar10ModelManager, CNNCifar100ModelManager]
    ):
        """Initialize client attributes.

        Args:
            model_manager: the model manager object
        """
        super().__init__()
        self.model_manager = model_manager

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters."""
        return self.model_manager.model.get_parameters()

    def set_parameters(self, parameters: NDArrays, evaluate: bool = False) -> None:
        """Set the local model parameters to the received parameters.

        Args:
            parameters: parameters to set the model to.
            evaluate: whether to evaluate or not.
        """
        _ = evaluate
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body") or k.startswith("_head")
        ]
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model_manager.model.set_parameters(state_dict)

    def perform_train(self) -> Dict[str, Union[List[Dict[str, float]], int, float]]:
        """Perform local training to the whole model.

        Returns
        -------
            Dict with the train metrics.
        """
        self.model_manager.model.enable_body()
        self.model_manager.model.enable_head()

        return self.model_manager.train(self.device)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Train the provided parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
            Tuple containing the locally updated model parameters, \
                the number of examples used for training and \
                the training metrics.
        """
        self.set_parameters(parameters)
        self.perform_train()

        return self.get_parameters(config), self.model_manager.train_dataset_size(), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Union[bool, bytes, float, int, str]]]:
        """Evaluate the provided global parameters using the locally held dataset.

        Args:
            parameters: The current (global) model parameters.
            config: configuration parameters for training sent by the server.

        Returns
        -------
        Tuple containing the test loss, \
                the number of examples used for evaluation and \
                the evaluation metrics.
        """
        self.set_parameters(parameters, evaluate=True)

        # Test the model
        test_results = self.model_manager.test(self.device)

        return (
            test_results.get("loss", 0.0),
            self.model_manager.test_dataset_size(),
            {k: v for k, v in test_results.items() if not isinstance(v, (dict, list))},
        )


class FedRepClient(BaseClient):
    """Implementation of Federated Personalization (FedRep) Client."""

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local body parameters."""
        return [
            val.cpu().numpy()
            for val in self.model_manager.model.body.state_dict().values()
        ]

    def set_parameters(self, parameters: NDArrays, evaluate: bool = False) -> None:
        """Set the local body parameters to the received parameters.

        Args:
            parameters: parameters to set the body to.
            evaluate: whether the client is evaluating or not.
        """
        model_keys = [
            k
            for k in self.model_manager.model.state_dict().keys()
            if k.startswith("_body")
        ]

        if not evaluate:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend(
                [
                    k
                    for k in self.model_manager.model.state_dict().keys()
                    if k.startswith("_head")
                ]
            )

        state_dict = OrderedDict(
            (k, torch.from_numpy(v)) for k, v in zip(model_keys, parameters)
        )

        self.model_manager.model.set_parameters(state_dict)


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    model_manager_class = get_model_manager_class(context)
    algorithm = str(context.run_config["algorithm"]).lower()
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader = load_data(
        partition_id, num_partitions, context
    )  # load the data
    if algorithm == Algorithm.FEDAVG.value:
        client_class = BaseClient
    elif algorithm == Algorithm.FEDREP.value:
        # This state variable will only be used by the FedRep algorithm.
        # We only need to initialize once, since client_fn will be called
        # again at every invocation of the ClientApp.
        if FEDREP_HEAD_STATE not in context.state.parameters_records:
            context.state.parameters_records[FEDREP_HEAD_STATE] = ParametersRecord()
        client_class = FedRepClient
    else:
        raise RuntimeError(f"Unknown algorithm {algorithm}.")

    model_manager_obj = model_manager_class(
        context=context, trainloader=trainloader, testloader=valloader
    )

    # Return client object.
    client = client_class(model_manager_obj).to_client()
    return client


# Flower ClientApp
app = ClientApp(client_fn)
