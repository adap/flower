"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import flwr as fl
import torch
import numpy as np
import torch.nn as nn

from typing import Callable, Dict, List, Tuple, Union
from omegaconf import DictConfig
from collections import OrderedDict
from hydra.utils import instantiate
from FedPer.utils import ModelManager
from FedPer.models import test, train, ModelManager, ModelSplit, MobileNet_v1
from torch.utils.data import DataLoader
from flwr.common.typing import NDArrays, Scalar

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
    ):  # pylint: disable=too-many-arguments
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.train_id = 1

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        if self.net.split: 
            return [val.cpu().numpy() for _, val in self.net.body.state_dict().items()]
        else:
            return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    
    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set the local body parameters to the received parameters.
        In the first train round the head parameters are also set to the global head parameters,
        to ensure every client head is initialized equally.

        Args:
            parameters: parameters to set the body to.
        """
        # Get model keys for body
        model_keys = [k for k in self.net.state_dict().keys() if k.startswith("body")]
        # print("Model keys: ", model_keys)
        if self.train_id == 1:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend([k for k in self.net.state_dict().keys() if k.startswith("head")])
        # Zip model keys and parameters
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # self.net.set_parameters(state_dict)
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        # Set parameters 
        self.set_parameters(parameters)
        print("config: ", config)
        # Epochs
        epochs = config["epochs"]

        # Train model
        train(
            net=self.net,
            trainloader=self.trainloader,
            device=self.device,
            epochs=epochs,
            learning_rate=self.learning_rate,
        )

        self.train_id += 1

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def gen_client_fn(
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    model: DictConfig,
) -> Tuple[
    Callable[[str], FlowerClient], DataLoader
]:  # pylint: disable=too-many-arguments
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
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
    model : DictConfig
        The model configuration.

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
        net = instantiate(model).to(device)

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
        )
    
    return client_fn