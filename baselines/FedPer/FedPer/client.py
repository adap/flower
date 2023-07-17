"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from FedPer.dataset import load_datasets
from FedPer.models import test, train

import copy
import torch
import torch.nn as nn
import numpy as np
import time

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

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return [val.cpu().numpy() for _, val in self.model_manager.model.body.state_dict().items()]
    

    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Set the local body parameters to the received parameters.
        In the first train round the head parameters are also set to the global head parameters,
        to ensure every client head is initialized equally.

        Args:
            parameters: parameters to set the body to.
        """
        # Get model keys for body
        model_keys = [k for k in self.model_manager.model.state_dict().keys() if k.startswith("_body")]

        if self.train_id == 1:
            # Only update client's local head if it hasn't trained yet
            model_keys.extend([k for k in self.model_manager.model.state_dict().keys() if k.startswith("_head")])

        # Zip model keys and parameters
        params_dict = zip(model_keys, parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model_manager.model.set_parameters(state_dict)

    def fit(
        self, 
        parameters: NDArrays, 
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        # Set parameters
        self.set_parameters(parameters)

        # Set number of epochs
        num_epochs = self.num_epochs

        # Train model
        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=num_epochs,
            learning_rate=self.learning_rate,
        )

        return self.get_parameters({}), len(self.trainloader), {}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

def filter_cifar10(dataloader: DataLoader, num_classes: int) -> DataLoader:
    """Filters out all classes except the first num_classes classes.

    Parameters
    ----------
    dataloader : DataLoader
        The dataloader to filter.
    num_classes : int
        The number of classes to keep.

    Returns
    -------
    DataLoader
        The filtered dataloader.
    """
    # Filter out all classes except the first num_classes classes
    dataloader.dataset.targets = np.array(dataloader.dataset.targets)
    print(dataloader.dataset.targets)
    quit()
    mask = np.isin(dataloader.dataset.targets, list(range(num_classes)))
    dataloader.dataset.targets = dataloader.dataset.targets[mask].tolist()
    dataloader.dataset.data = dataloader.dataset.data[mask].tolist()

    return dataloader

def gen_client_fn(
    num_clients: int,
    num_rounds: int,
    num_epochs: int,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    learning_rate: float,
    model: DictConfig,
    num_classes: int,
    dataset_name : str
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
    model : DictConfig
        The model configuration.
    num_classes : int
        The number of classes in the dataset.
    dataset_name : str 
        The name of the dataset.

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

        if dataset_name == 'cifar10':
            if num_classes != 10:
                # only include num_classes classes in the training and validation set
                trainloader = filter_cifar10(trainloader, num_classes)
                valloader = filter_cifar10(valloader, num_classes)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented")

        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
        )

    return client_fn