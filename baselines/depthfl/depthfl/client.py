"""Defines the DepthFL Flower Client and a function to instantiate it."""

import copy
import pickle
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.client import Client
from flwr.client.numpy_client import NumPyClient
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Code, NDArrays, Scalar, Status
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from depthfl.models import test, train


def prune(state_dict, param_idx):
    """Prune width of DNN (for HeteroFL)."""
    ret_dict = {}
    for k in state_dict.keys():
        if "num" not in k:
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
        prev_grads: Dict,
        cid: int,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.prev_grads = prev_grads
        self.cid = cid
        self.param_idx = {}
        state_dict = net.state_dict()

        # for HeteroFL
        for k in state_dict.keys():
            self.param_idx[k] = [
                torch.arange(size) for size in state_dict[k].shape
            ]  # store client's weights' shape (for HeteroFL)

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(prune(state_dict, self.param_idx), strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
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
            learning_rate=self.learning_rate * self.learning_rate_decay**curr_round,
            feddyn=config["feddyn"],
            kd=config["kd"],
            consistency_weight=consistency_weight,
            prev_grads=self.prev_grads,
            alpha=config["alpha"],
            extended=config["extended"],
        )  

        with open(f'prev_grads/client_{self.cid}', 'wb') as f:
            pickle.dump(self.prev_grads, f)

        return self.get_parameters({}), len(self.trainloader), {"cid": self.cid}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy, accuracy_single = test(self.net, self.valloader, self.device)
        return (
            float(loss),
            len(self.valloader),
            {"accuracy": float(accuracy), "accuracy_single": accuracy_single},
        )


def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    learning_rate_decay: float,
    models: List[DictConfig],
    cfg: DictConfig,
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

        with open(f'prev_grads/client_{int(cid)}', 'rb') as f:
            prev_grads = pickle.load(f)

        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            learning_rate_decay,
            prev_grads,
            int(cid),
        )

    return client_fn


