"""Define your client class and a function to construct such clients.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple
from omegaconf import DictConfig
from hydra.utils import instantiate

import flwr as fl
from flwr.common.typing import NDArrays, Scalar

import torch
import torch.nn
from torch.utils.data import DataLoader

from models import train, test

from dataset import  FemnistDataset
import torchvision.transforms as transforms


class FlowerClient(
    fl.client.NumPyClient
):
    def __init__(
            self,
            net: torch.nn.Module,
            trainloader: DataLoader,
            valloader: DataLoader,
            device: torch.device,
            num_epochs: int,
            learning_rate: float
    ) -> object:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

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
        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=self.num_epochs,
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


def gen_client_fn(
        num_epochs: int,
        dataset: Tuple[Dict],
        client_list: List[str],
        learning_rate: float,
        model: DictConfig,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
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

    Returns
    -------
    Tuple[Callable[[str], FlowerClient], DataLoader]
        A tuple containing the client function that creates Flower Clients and
        the DataLoader that will be used for testing
    """

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        print(f'cid : {cid}')

        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)
        transform = transforms.Compose([transforms.ToTensor()])

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        sup_set, qry_set = dataset
        train_clients, val_client, _ = client_list

        train_set = sup_set['user_data'][train_clients[int(cid)]]
        valid_set = sup_set['user_data'][val_client[int(cid)]]

        trainloader = DataLoader(FemnistDataset(train_set, transform), batch_size=10, shuffle=True)
        valloader = DataLoader(FemnistDataset(valid_set, transform))

        return FlowerClient(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
        )

    return client_fn
