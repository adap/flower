# pylint: disable=too-many-arguments
"""Defines the MNIST Flower Client and a function to instantiate it."""


from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from torch.utils.data import DataLoader

from flwr_baselines.publications.fedprox_mnist import model
from flwr_baselines.publications.fedprox_mnist.dataset import load_datasets


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        proximal_mu: float,
        staggler_schedule: np.ndarray,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.proximal_mu = proximal_mu
        self.staggler_schedule = staggler_schedule

    def get_parameters(self, config) -> List[np.ndarray]:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, int]
    ) -> Tuple[List[np.ndarray], int, dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)

        if self.staggler_schedule[config["curr_round"] - 1]:
            num_epochs = np.random.randint(1, self.num_epochs)
        else:
            num_epochs = self.num_epochs

        model.train(
            self.net,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
            proximal_mu=self.proximal_mu,
        )

        return self.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, int]):
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = model.test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    device: torch.device,
    iid: bool,
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    proximal_mu: float,
    stagglers: float,
) -> Tuple[Callable[[str], FlowerClient], DataLoader]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
    device : torch.device
        The device on which the the client will train on and test on.
    iid : bool
        The way to partition the data for each client, i.e. whether the data
        should be independent and identically distributed between the clients
        or if the data should first be sorted by labels and distributed by chunks
        to each client (used to test the convergence in a worst case scenario)
    num_clients : int
        The number of clients present in the setup
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.
    proximal_mu : float
        Parameter for the weight of the proximal term.
    stagglers : float
        Proportion of stagglers in the clients, between 0 and 1.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    trainloaders, valloaders, testloader = load_datasets(
        iid=iid, num_clients=num_clients, batch_size=batch_size
    )

    stagglers_mat = np.transpose(
        np.random.choice(
            [0, 1], size=(num_rounds, num_clients), p=[1 - stagglers, stagglers]
        )
    )

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        net = model.Net().to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
            proximal_mu,
            stagglers_mat[int(cid)],
        )

    return client_fn, testloader
