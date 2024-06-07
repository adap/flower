"""Clients implementation for Flanders."""

from collections import OrderedDict
from pathlib import Path
from typing import Tuple

import flwr as fl
import numpy as np
import ray
import torch

from .dataset import get_dataloader, mnist_transformation
from .models import (
    FMnistNet,
    MnistNet,
    test_fmnist,
    test_mnist,
    train_fmnist,
    train_mnist,
)

XY = Tuple[np.ndarray, np.ndarray]


def get_params(model):
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model, params):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class MnistClient(fl.client.NumPyClient):
    """Implementation of MNIST image classification using PyTorch."""

    def __init__(self, cid, fed_dir_data):
        """Instantiate a client for the MNIST dataset."""
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = MnistNet()

        # Determine device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print("Device: ", self.device)

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy ndarrays."""
        return get_params(self.net)

    def fit(self, parameters, config):
        """Set model parameters from a list of NumPy ndarrays."""
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 1
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            transform=mnist_transformation,
        )

        self.net.to(self.device)
        train_mnist(self.net, trainloader, epochs=config["epochs"], device=self.device)

        return (
            get_params(self.net),
            len(trainloader.dataset),
            {"cid": self.cid, "malicious": config["malicious"]},
        )

    def evaluate(self, parameters, config):
        """Evaluate using local test dataset."""
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=False,
            batch_size=50,
            workers=num_workers,
            transform=mnist_transformation,
        )

        self.net.to(self.device)
        loss, accuracy = test_mnist(self.net, valloader, device=self.device)

        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


class FMnistClient(fl.client.NumPyClient):
    """Implementation of MNIST image classification using PyTorch."""

    def __init__(self, cid, fed_dir_data):
        """Instantiate a client for the MNIST dataset."""
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = FMnistNet()

        # Determine device
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print("Device: ", self.device)

    def get_parameters(self, config):
        """Get model parameters as a list of NumPy ndarrays."""
        return get_params(self.net)

    def fit(self, parameters, config):
        """Set model parameters from a list of NumPy ndarrays."""
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = 1
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            transform=mnist_transformation,
        )

        self.net.to(self.device)
        train_fmnist(self.net, trainloader, epochs=config["epochs"], device=self.device)

        return (
            get_params(self.net),
            len(trainloader.dataset),
            {"cid": self.cid, "malicious": config["malicious"]},
        )

    def evaluate(self, parameters, config):
        """Evaluate using local test dataset."""
        set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=False,
            batch_size=50,
            workers=num_workers,
            transform=mnist_transformation,
        )

        self.net.to(self.device)
        loss, accuracy = test_fmnist(self.net, valloader, device=self.device)

        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
