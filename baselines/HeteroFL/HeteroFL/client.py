"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
# pylint: disable=too-many-arguments
"""Defines the MNIST Flower Client and a function to instantiate it."""


from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
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
from flwr.common.typing import NDArrays, Scalar
from models import get_parameters, set_parameters, test, train
from torch.utils.data import DataLoader


class FlowerNumPyClient(fl.client.NumPyClient):
    def __init__(
        self, cid, net, trainloader, label_split, valloader, epochs, model_rate, device = "cpu"
    ):
        self.cid = cid
        # self.net = conv(model_rate = model_rate)
        self.net = net
        self.trainloader = trainloader
        self.label_split = label_split
        self.valloader = valloader
        self.epochs = epochs
        self.model_rate = model_rate
        self.device = device
        print(
            "Client_with model rate = {} , cid of client = {}".format(
                self.model_rate, self.cid
            )
        )

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print("cid = {}".format(self.cid))
        set_parameters(self.net, parameters)
        train(
            self.net,
            self.trainloader,
            self.label_split,
            epochs=self.epochs,
            lr=config["lr"],
            device=self.device,
        )
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        # print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    model,
    data_shape,
    hidden_layers,
    classes_size,
    norm,
    global_model_rate,
    num_clients,
    client_to_model_rate_mapping,
    num_epochs,
    trainloaders,
    label_split,
    valloaders,
    device="cpu",
) -> Tuple[
    Callable[[str], FlowerNumPyClient], DataLoader
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

    # Defines a staggling schedule for each clients, i.e at which round will they
    # be a straggler. This is done so at each round the proportion of staggling
    # clients is respected

    def client_fn(cid: str) -> FlowerNumPyClient:
        """Create a Flower client representing a single organization."""

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        model_rate = client_to_model_rate_mapping[int(cid)]

        return FlowerNumPyClient(
            cid=cid,
            net=model(
                model_rate=model_rate,
                data_shape=data_shape,
                hidden_layers=hidden_layers,
                classes_size=classes_size,
                norm = norm,
                global_model_rate=global_model_rate,
            ).to(device),
            trainloader=trainloader,
            label_split=label_split[int(cid)],
            valloader=valloader,
            epochs=num_epochs,
            model_rate=model_rate,
        )

    return client_fn
