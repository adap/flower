"""Defines the MNIST Flower Client and a function to instantiate it."""
import pickle
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import os
import flwr as fl
import copy

import torch
from utils import apply_nn_compression
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from tamuna.models import train


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        device: torch.device,
        learning_rate: float,
        cid: str,
    ):
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.learning_rate = learning_rate
        self.control_variate = None
        self.old_compression_mask = None
        self.old_compressed_net = None
        self.cid = cid

        self.state_file_name = f"{self.cid}_state.bin"
        self.mask_file_name = f"{self.cid}_mask.bin"
        self.__create_state(self.state_file_name)

    def __create_state(self, state_file_name):
        if not os.path.exists(state_file_name):
            with open(state_file_name, "wb") as f:
                state = (
                    self.control_variate,
                    self.old_compression_mask,
                    self.old_compressed_net,
                )
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(f"Client {self.cid} state created.")
        else:
            print(f"Client {self.cid} state already exists.")

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
    ) -> Tuple[NDArrays, int, Dict[str, int]]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)

        with open(self.mask_file_name, "rb") as f:
            mask = pickle.load(f)

        self.__load_state()

        self.net, self.control_variate = train(
            self.net,
            self.trainloader,
            self.device,
            epochs=config["epochs"],
            eta=config["eta"],
            control_variate=self.control_variate,
            old_compression_mask=self.old_compression_mask,
            old_compressed_net=self.old_compressed_net,
            server_net=copy.deepcopy(self.net),
            lr=self.learning_rate,
        )

        self.net = apply_nn_compression(self.net, mask)

        self.__save_state(mask)

        return self.get_parameters({}), len(self.trainloader), {}

    def __save_state(self, mask):
        with open(self.state_file_name, "wb") as f:
            state = (
                self.control_variate,
                mask,
                self.net,
            )
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def __load_state(self):
        with open(self.state_file_name, "rb") as f:
            state = pickle.load(f)
            (
                self.control_variate,
                self.old_compression_mask,
                self.old_compressed_net,
            ) = state


def gen_client_fn(
    trainloaders: List[DataLoader],
    learning_rate: float,
    model: DictConfig,
    client_device: str,
) -> Callable[[str], FlowerClient]:
    """Generates the client function that creates the Flower Clients.

    Parameters
    ----------
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

        print(f"Generating client {cid}.")

        # Load model
        device = torch.device(device=client_device)
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader, so each client
        # will train on their own unique data
        trainloader = trainloaders[int(cid)]

        return FlowerClient(net, trainloader, device, learning_rate, cid)

    return client_fn
