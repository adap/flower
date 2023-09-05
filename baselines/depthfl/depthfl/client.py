"""Defines the DepthFL Flower Client and a function to instantiate it."""

import copy
import torch
import numpy as np
import flwr as fl
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Union
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common.typing import NDArrays, Scalar, Status, Code
from flwr.client import Client
from flwr.client.app import (
    numpyclient_has_get_properties,
    numpyclient_has_get_parameters,
    numpyclient_has_fit,
    numpyclient_has_evaluate,
    _get_properties,
    _get_parameters,
    _evaluate,
    _constructor,
)
from flwr.client.numpy_client import NumPyClient

from depthfl.models import test, train
from depthfl import FitIns, FitRes

EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT = """
NumPyClient.fit did not return a tuple with 3 elements.
The returned values should have the following type signature:

    Tuple[NDArrays, Dict, int]
"""

ClientLike = Union[Client, NumPyClient]

def prune(state_dict, param_idx):
    """prune width of DNN (for HeteroFL)"""

    ret_dict = {}
    for k in state_dict.keys():
        if 'num' not in k:
            ret_dict[k] = state_dict[k][torch.meshgrid(param_idx[k])]
        else:
            ret_dict[k] = state_dict[k]
    return copy.deepcopy(ret_dict)

class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        learning_rate_decay: float,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.param_idx = {}
        state_dict = net.state_dict()

        # for HeteroFL
        for k in state_dict.keys():
            self.param_idx[k] = [torch.arange(size) for size in state_dict[k].shape] # store client's weights' shape (for HeteroFL)
        

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(prune(state_dict, self.param_idx), strict=True) 

    def fit(
        self, parameters: NDArrays, prev_grads: Dict, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, Dict, int]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        num_epochs = self.num_epochs

        curr_round = config["curr_round"] - 1

        # consistency weight for self distillation in DepthFL
        CONSISTENCY_WEIGHT = 300
        current = np.clip(curr_round, 0.0, CONSISTENCY_WEIGHT)
        phase = 1.0 - current / CONSISTENCY_WEIGHT
        consistency_weight = float(np.exp(-5.0 * phase * phase))

        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate * self.learning_rate_decay ** curr_round,
            feddyn=config["feddyn"],
            kd=config["kd"],
            consistency_weight=consistency_weight,
            prev_grads = prev_grads,
            alpha=config["alpha"],
            extended=config["extended"],
        )

        return self.get_parameters({}), prev_grads, len(self.trainloader)

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy, accuracy_single = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy), "accuracy_single":accuracy_single}


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    learning_rate_decay: float,
    models: List[DictConfig],
    cfg: DictConfig
) -> Tuple[
    Callable[[str], FlowerClient], DataLoader
]:  # pylint: disable=too-many-arguments
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    num_clients : int
        The number of clients present in the setup
    num_rounds: int
        The number of rounds in the experiment. This is used to construct
        the scheduling for stragglers
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    learning_rate_decay : float
        The learning rate decay ratio per round for the SGD  optimizer of clients.
    models : List[DictConfig]
        A list of DictConfigs, each pointing to the model config of client's local model

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # each client gets a different model config (different width / depth)
        net = instantiate(models[int(cid)]).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        
        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            learning_rate_decay,
        )

    return client_fn



def _fit(self: Client, ins: FitIns) -> FitRes:

    """Refine the provided parameters using the locally held dataset.
    FitIns & FitRes were modified for FedDyn. Fit function gets prev_grads 
    as input and return the updated prev_grads with updated parameters
    """

    # Deconstruct FitIns
    parameters: NDArrays = parameters_to_ndarrays(ins.parameters)

    # Train
    results = self.numpy_client.fit(parameters, ins.prev_grads, ins.config)  # type: ignore
    if not (
        len(results) == 3
        and isinstance(results[0], list)
        and isinstance(results[1], Dict)
        and isinstance(results[2], int)
    ):
        raise Exception(EXCEPTION_MESSAGE_WRONG_RETURN_TYPE_FIT)

    # Return FitRes
    parameters_prime, prev_grads, num_examples = results
    parameters_prime_proto = ndarrays_to_parameters(parameters_prime)
    return FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=parameters_prime_proto,
        prev_grads=prev_grads,
        num_examples=num_examples,
        cid = -1,
    )


def _wrap_numpy_client(client: NumPyClient) -> Client:
    member_dict: Dict[str, Callable] = {  # type: ignore
        "__init__": _constructor,
    }

    # Add wrapper type methods (if overridden)

    if numpyclient_has_get_properties(client=client):
        member_dict["get_properties"] = _get_properties

    if numpyclient_has_get_parameters(client=client):
        member_dict["get_parameters"] = _get_parameters

    if numpyclient_has_fit(client=client):
        member_dict["fit"] = _fit

    if numpyclient_has_evaluate(client=client):
        member_dict["evaluate"] = _evaluate

    # Create wrapper class
    wrapper_class = type("NumPyClientWrapper", (Client,), member_dict)

    # Create and return an instance of the newly created class
    return wrapper_class(numpy_client=client)  # type: ignore

def to_client(client_like: ClientLike) -> Client:
    """Take any Client-like object and return it as a Client."""
    if isinstance(client_like, NumPyClient):
        return _wrap_numpy_client(client=client_like)
    return client_like