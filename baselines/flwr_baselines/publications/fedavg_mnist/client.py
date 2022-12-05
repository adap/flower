"""Defines the MNIST Flower Client and a function to instantiate it."""
from collections import OrderedDict
from typing import List, Tuple

import dataset
import flwr as fl
import model
import numpy as np
import torch
from torch.utils.data import DataLoader

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self, net: torch.nn.Module, trainloader: DataLoader, valloader: DataLoader
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config
    ) -> Tuple[List[np.ndarray], int, dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        model.train(self.net, self.trainloader, DEVICE, epochs=1)
        return self.get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters: List[np.ndarray], config):
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        loss, accuracy = model.test(self.net, self.valloader, DEVICE)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = model.Net().to(DEVICE)

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloaders, valloaders, _ = dataset.load_datasets(idd=False)
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader)
