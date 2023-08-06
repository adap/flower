"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
# pylint: disable=too-many-arguments
"""Defines the MNIST Flower Client and a function to instantiate it."""


from collections import OrderedDict
from typing import Callable, Dict, Tuple, List

import flwr as fl
import torch
import numpy as np
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from HeterFL.models import test , train, set_parameters , get_parameters

from flwr_baselines.publications.fedavg_mnist import model
from flwr_baselines.publications.fedavg_mnist.dataset import load_datasets


class FlowerClient(fl.client.Client):
    def __init__(self, cid, net, trainloader, valloader, model_rate):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.model_rate = model_rate

    def get_properties(self , ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(Status(code=Code.OK , message= "Success") , {'client_model_rate' : self.model_rate})


    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        print(f"[Client {self.cid}] get_parameters")
        # Get parameters as a list of NumPy ndarray's
        ndarrays: List[np.ndarray] = get_parameters(self.net)

        # Serialize ndarray's into a Parameters object
        parameters = ndarrays_to_parameters(ndarrays)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return GetParametersRes(
            status=status,
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"[Client {self.cid}] fit, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        # Update local model, train, get updated parameters
        set_parameters(self.net, ndarrays_original)
        train(self.net, self.trainloader, epochs=1)
        ndarrays_updated = get_parameters(self.net)

        # Serialize ndarray's into a Parameters object
        parameters_updated = ndarrays_to_parameters(ndarrays_updated)

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return FitRes(
            status=status,
            parameters=parameters_updated,
            num_examples=len(self.trainloader),
            metrics={},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        print(f"[Client {self.cid}] evaluate, config: {ins.config}")

        # Deserialize parameters to NumPy ndarray's
        parameters_original = ins.parameters
        ndarrays_original = parameters_to_ndarrays(parameters_original)

        set_parameters(self.net, ndarrays_original)
        loss, accuracy = test(self.net, self.valloader)
        # return float(loss), len(self.valloader), {"accuracy": float(accuracy)}

        # Build and return response
        status = Status(code=Code.OK, message="Success")
        return EvaluateRes(
            status=status,
            loss=float(loss),
            num_examples=len(self.valloader),
            metrics={"accuracy": float(accuracy)},
        )


def gen_client_fn(
    device: torch.device,
    iid: bool,
    balance: bool,
    num_clients: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    model_split_rate: list,
    model_split_mode,
    model_mode: dict[str, int],
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
    balance : bool
        Whether the dataset should contain an equal number of samples in each class,
        by default True
    num_clients : int
        The number of clients present in the setup
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    batch_size : int
        The size of the local batches each client trains on.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """
    trainloaders, valloaders, testloader = load_datasets(
        iid=iid, balance=balance, num_clients=num_clients, batch_size=batch_size
    )

    model_rate_of_clients = create_model_rate(model_split_rate , model_split_mode , model_mode)


    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        net = model.Net().to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        model_rate = model_rate_of_clients[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(
            net, trainloader, valloader, device, num_epochs, learning_rate, model_rate=model_rate
        )

    return client_fn, testloader