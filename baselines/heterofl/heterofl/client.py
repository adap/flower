"""Defines the MNIST Flower Client and a function to instantiate it."""

from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays

from heterofl.models import create_model, get_parameters, set_parameters, test, train

# from torch.utils.data import DataLoader


class FlowerNumPyClient(fl.client.NumPyClient):
    """Standard Flower client for training."""

    def __init__(
        self,
        # cid: str,
        net: torch.nn.Module,
        dataloader,
        model_rate: Optional[float],
        client_train_settings: Dict,
    ):
        # self.cid = cid
        self.net = net
        self.trainloader = dataloader["trainloader"]
        self.label_split = dataloader["label_split"]
        self.valloader = dataloader["valloader"]
        self.model_rate = model_rate
        self.client_train_settings = client_train_settings
        self.client_train_settings["device"] = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
        # print(
        #     "Client_with model rate = {} , cid of client = {}".format(
        #         self.model_rate, self.cid
        #     )
        # )

    def get_parameters(self, config) -> NDArrays:
        """Return the parameters of the current net."""
        # print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        # print(f"cid = {self.cid}")
        set_parameters(self.net, parameters)
        if "lr" in config:
            self.client_train_settings["lr"] = config["lr"]
        train(
            self.net,
            self.trainloader,
            self.label_split,
            self.client_train_settings,
        )
        return get_parameters(self.net), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        set_parameters(self.net, parameters)
        loss, accuracy = test(
            self.net, self.valloader, device=self.client_train_settings["device"]
        )
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    model_config: Dict,
    client_to_model_rate_mapping: Optional[List[float]],
    client_train_settings: Dict,
    data_loaders,
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

    Returns
    -------
    Callable[[str], FlowerClient]
        A tuple containing the client function that creates Flower Clients
    """

    def client_fn(cid: str) -> FlowerNumPyClient:
        """Create a Flower client representing a single organization."""
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        client_dataloader = {
            "trainloader": data_loaders["trainloaders"][int(cid)],
            "valloader": data_loaders["valloaders"][int(cid)],
            "label_split": data_loaders["label_split"][int(cid)],
        }
        # trainloader = data_loaders["trainloaders"][int(cid)]
        # valloader = data_loaders["valloaders"][int(cid)]
        model_rate = None
        if client_to_model_rate_mapping is not None:
            model_rate = client_to_model_rate_mapping[int(cid)]

        return FlowerNumPyClient(
            # cid=cid,
            net=create_model(
                model_config,
                model_rate=model_rate,
                device=device,
            ),
            dataloader=client_dataloader,
            model_rate=model_rate,
            client_train_settings=client_train_settings,
        )

    return client_fn
