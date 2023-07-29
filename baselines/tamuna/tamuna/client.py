"""Defines the MNIST Flower Client and a function to instantiate it."""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import copy
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tamuna.models import test, train


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        learning_rate: float,
    ):
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.learning_rate = learning_rate
        self.control_variate = self.__model_zeroed_out()

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def __model_zeroed_out(self):
        control_variate = copy.deepcopy(self.net)
        state_dict = OrderedDict(
            {k: torch.zeros_like(v) for k, v in self.net.state_dict().items()}
        )
        control_variate.load_state_dict(state_dict, strict=True)
        return control_variate

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, int]]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)

        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=config["epochs"],
            learning_rate=self.learning_rate,
        )

        return self.get_parameters({}), len(self.trainloader), {}


def gen_client_fn(
    trainloaders: List[DataLoader],
    learning_rate: float,
    model: DictConfig,
    client_device: str,
) -> Callable[[str], FlowerClient]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        device = torch.device(device=client_device)
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader, so each client
        # will train on their own unique data
        trainloader = trainloaders[int(cid)]

        return FlowerClient(net, trainloader, device, learning_rate)

    return client_fn
