"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""
import flwr as fl
from flwr.common import Scalar
from flwr.common.typing import Parameters
from typing import Dict, OrderedDict, Union, List, Tuple, Callable
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from flwr.common.logger import log
from logging import DEBUG, INFO

from scaffold.models import train, test


class FlowerClientScaffold(
    fl.client.NumPyClient
):
    """Flower client implementing scaffold."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
    ) -> None:
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        # initialize client control variate with 0 and shape of the network parameters
        self.client_cv = []
        for param in self.net.parameters():
            self.client_cv.append(torch.zeros(param.shape))
    
    def get_parameters(self, config: Dict[str, Scalar]):
        """Return the current local model parameters."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        """Set the local model parameters using given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, config: Dict[str, Union[Scalar, List[torch.Tensor]]]):
        """Implements distributed fit function for a given client using Scaffold Strategy."""
        self.set_parameters(parameters)
        # server_cv and client_cv are zero
        server_cv = self.client_cv
        train(
            self.net,
            self.trainloader,
            self.device,
            self.num_epochs,
            self.learning_rate,
            server_cv,
            self.client_cv,
        )
        return self.get_parameters({}), len(self.trainloader), {}
    
    def evaluate(self, parameters, config: Dict[str, Scalar]):
        """Evaluate using given parameters."""
        self.set_parameters(parameters)
        loss, acc = test(self.net, self.valloader, self.device)
        return float(loss), len(self.valloader.dataset), {"accuracy": float(acc)}

def gen_client_fn(
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    num_epochs: int,
    learning_rate: float,
    model: DictConfig,
) -> Tuple[
    Callable[[str], FlowerClientScaffold], DataLoader
]:  # pylint: disable=too-many-arguments
    """Generates the client function that creates the scaffold flower clients.

    Parameters
    ----------
    trainloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset training partition
        belonging to a particular client.
    valloaders: List[DataLoader]
        A list of DataLoaders, each pointing to the dataset validation partition
        belonging to a particular client.
    num_epochs : int
        The number of local epochs each client should run the training for before
        sending it to the server.
    learning_rate : float
        The learning rate for the SGD  optimizer of clients.

    Returns
    -------
    Tuple[Callable[[str], FlowerClientScaffold], DataLoader]
        A tuple containing the client function that creates the scaffold flower clients and
        the DataLoader that will be used for testing
    """

    def client_fn(cid: str) -> FlowerClientScaffold:
        """Create a Flower client representing a single organization."""

        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]

        return FlowerClientScaffold(
            net,
            trainloader,
            valloader,
            device,
            num_epochs,
            learning_rate,
        )

    return client_fn