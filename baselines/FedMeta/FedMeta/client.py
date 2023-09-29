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

from models import train, test, train_meta, test_meta

class FlowerClient(
    fl.client.NumPyClient
):
    def __init__(
            self,
            net: torch.nn.Module,
            trainloaders: DataLoader,
            valloaders: DataLoader,
            cid: str,
            device: torch.device,
            num_epochs: int,
            learning_rate: float,
            gradient_step: int
    ) -> object:
        self.net = net
        self.trainloaders = trainloaders
        self.valloaders = valloaders
        self.cid = int(cid)
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.gradient_step = gradient_step

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
        algo = config["algo"]
        total_len = len(self.trainloaders['qry'][self.cid].dataset) + len(self.trainloaders['sup'][self.cid].dataset)
        if algo == 'fedavg' or algo == 'fedavg(meta)':
            loss = train(
                self.net,
                self.trainloaders['sup'][self.cid],
                self.trainloaders['qry'][self.cid],
                self.device,
                epochs=self.num_epochs,
                learning_rate=self.learning_rate
            )
            return self.get_parameters({}), total_len, {"loss" : loss}

        elif algo == 'fedmeta(maml)' or algo == 'fedmeta(meta-sgd)':
            alpha = config["alpha"]
            loss, grads = train_meta(
                self.net,
                self.trainloaders['sup'][self.cid],
                self.trainloaders['qry'][self.cid],
                alpha,
                self.device,
                self.gradient_step,
            )
            return self.get_parameters({}), total_len, {"loss": loss, "grads": grads}

    def evaluate(
            self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implements distributed evaluation for a given client."""
        self.set_parameters(parameters)
        total_len = len(self.valloaders['qry'][self.cid].dataset) + len(self.valloaders['sup'][self.cid].dataset)
        if config["algo"] == 'fedavg' or config["algo"] == 'fedavg(meta)':
            loss, accuracy, total = test(
                    self.net,
                    self.valloaders['sup'][self.cid],
                    self.valloaders['qry'][self.cid],
                    self.device,
                    config["algo"],
                    config["data"],
                    learning_rate=self.learning_rate,
            )
            return float(loss), total_len, {"correct": accuracy, "loss": loss}

        elif config["algo"] == 'fedmeta(maml)' or config["algo"] == 'fedmeta(meta-sgd)':
            alpha = config["alpha"]
            loss, accuracy, total = test_meta(
                self.net,
                self.valloaders['sup'][self.cid],
                self.valloaders['qry'][self.cid],
                alpha,
                self.device,
                self.gradient_step,
            )
            return float(loss), total_len, {"correct": float(accuracy), "loss": loss}


def gen_client_fn(
        num_epochs: int,
        trainloaders: List[DataLoader],
        valloaders: List[DataLoader],
        learning_rate: float,
        model: DictConfig,
        gradient_step: int,
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
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)

        return FlowerClient(
            net,
            trainloaders,
            valloaders,
            cid,
            device,
            num_epochs,
            learning_rate,
            gradient_step
        )

    return client_fn
