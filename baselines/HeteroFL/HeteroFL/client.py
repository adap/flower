"""Defines the MNIST Flower Client and a function to instantiate it."""

from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays
from models import create_model, get_parameters, set_parameters, test, train
from torch.utils.data import DataLoader


class FlowerNumPyClient(fl.client.NumPyClient):
    """Standard Flower client for training."""

    def __init__(
        self,
        cid: str,
        net: torch.nn.Module,
        trainloader: DataLoader,
        label_split: torch.tensor,
        valloader: DataLoader,
        model_rate: float,
        client_train_settings: Dict,
    ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.label_split = label_split
        self.valloader = valloader
        self.model_rate = model_rate
        self.client_train_settings = client_train_settings
        # print(
        #     "Client_with model rate = {} , cid of client = {}".format(
        #         self.model_rate, self.cid
        #     )
        # )

    def get_parameters(self, config) -> NDArrays:
        """Return the parameters of the current net."""
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        print("cid = {}".format(self.cid))
        set_parameters(self.net, parameters)
        self.client_train_settings["lr"] = config["lr"]
        train(
            self.net,
            self.trainloader,
            self.label_split,
            self.client_train_settings,
        )
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        set_parameters(self.net, parameters)
        loss, accuracy = test(
            self.net, self.valloader, device=self.client_train_settings["device"]
        )
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    model_config: Dict,
    client_to_model_rate_mapping: List[float],
    client_train_settings: Dict,
    trainloaders: List[DataLoader],
    label_split: torch.tensor,
    valloaders: List[DataLoader],
    device="cpu",
) -> Callable[[str], FlowerNumPyClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients.

    Parameters
    ----------
    model_config : Dict
        Dict that contains all the information required to
        create a model (data_shape , hidden_layers , classes_size...)
    client_to_model_rate: List[float]
        List tha contains model_rates of clients.
        model_rate of client with cid i = client_to_model_rate_mapping[i]
    client_train_settings : Dict
        Dict that contains information regarding optimizer , lr ,
        momentum , device required by the client to train
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    label_split: torch.tensor
        A Tensor of tensors that conatins the labels of the partitioned dataset.
        label_split of client with cid i = label_split[i]
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    device : str
        Device the client need to train on.

    Returns
    -------
    Callable[[str], FlowerClient]
        A tuple containing the client function that creates Flower Clients
    """

    def client_fn(cid: str) -> FlowerNumPyClient:
        """Create a Flower client representing a single organization."""
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        model_rate = client_to_model_rate_mapping[int(cid)]

        return FlowerNumPyClient(
            cid=cid,
            net=create_model(
                model_config,
                model_rate=model_rate,
                device=device,
            ),
            trainloader=trainloader,
            label_split=label_split[int(cid)],
            valloader=valloader,
            model_rate=model_rate,
            client_train_settings=client_train_settings,
        )

    return client_fn
