"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
"""Defines the MNIST Flower Client and a function to instantiate it."""


from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import os


from moon.models import train_moon, train_fedprox, init_net


class FlowerClient(
    fl.client.NumPyClient
):  
    """Standard Flower client for CNN training."""
    def __init__(
        self,
        net: torch.nn.Module,
        net_id: int,
        dataset: str,
        model: str,
        output_dim: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        mu: float,
        temperature: float,
        model_dir: str,
        alg: str,
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.net_id = net_id
        self.dataset = dataset
        self.model = model
        self.output_dim = output_dim
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.mu = mu
        self.temperature = temperature
        self.model_dir = model_dir
        self.alg = alg
        

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        
        #load previous model from model_dir
        self.prev_net = init_net(self.dataset, self.model, self.output_dim)
        self.prev_net.load_state_dict(torch.load(os.path.join(self.model_dir, "prev_net.pt")))
        global_net = init_net(self.dataset, self.model, self.output_dim)
        global_net.load_state_dict(self.net.state_dict())
        if self.alg == "moon":
            train_moon(
                self.net,
                global_net,
                self.prev_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.temperature,
                self.device
            )
        elif self.alg == "fedprox":
            train_fedprox(
                self.net,
                global_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.device
            )
        torch.save(self.net.state_dict(), os.path.join(self.model_dir, "prev_net.pt"))
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    trainloaders: List[DataLoader],
    testloaders: List[DataLoader],
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
    stragglers : float
        Proportion of stragglers in the clients, between 0 and 1.

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
        net = init_net(cfg.dataset.name, cfg.model.name, cfg.model.output_dim)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        testloader = testloaders[int(cid)]

        return FlowerClient(
            net,
            int(cid),
            cfg.dataset.name,
            cfg.model.name,
            cfg.model.output_dim,
            trainloader,
            testloader,
            device,
            cfg.num_epochs,
            cfg.learning_rate,
            cfg.mu,
            cfg.temperature,
            cfg.model.dir,
            cfg.alg,
        )
    return client_fn
