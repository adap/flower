"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import os
from collections import OrderedDict
import copy
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from moon.models import init_net, test, train_fedprox, train_moon

# pylint: disable=E1101


# pylint: disable=too-many-instance-attributes
class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        # net: torch.nn.Module,
        net_id: int,
        dataset: str,
        model: str,
        output_dim: int,
        trainloader: DataLoader,
        valloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        mu: float,
        temperature: float,
        model_dir: str,
        alg: str,
    ):  # pylint: disable=too-many-arguments
        self.net = init_net(dataset, model, output_dim)
        self.net_id = net_id
        self.dataset = dataset
        self.model = model
        self.output_dim = output_dim
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.mu = mu  # pylint: disable=invalid-name
        self.temperature = temperature
        self.model_dir = model_dir
        self.alg = alg
        # self.prev_net = init_net(self.dataset, self.model, self.output_dim)
        self.prev_net = None

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""
        # print("self.net:", self.net.state_dict())
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Change the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        # print("params_dict:", params_dict)
        # state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        state_dict = OrderedDict({k: torch.from_numpy(v) for k, v in params_dict})
        # print("state_dict:", state_dict)
        # try:
        self.net.load_state_dict(state_dict, strict=True)
        # except:
        #     print("error in loading")
        #     print("params_dict:", params_dict)
        #     print("state_dict:", state_dict)
        #     exit(0)
        # self.net.load_state_dict(state_dict)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        self.set_parameters(parameters)
        if self.prev_net is None:
            self.prev_net = init_net(self.dataset, self.model, self.output_dim)
            self.prev_net = copy.deepcopy(self.net)
        else:
        # if os.path.exists(os.path.join(self.model_dir, str(self.net_id), "prev_net.pt")):
            # load previous model from model_dir
            self.prev_net.load_state_dict(
                torch.load(os.path.join(self.model_dir, str(self.net_id), "prev_net.pt"))
            )
        # else:
            # self.prev_net = copy.deepcopy(self.net)
        global_net = init_net(self.dataset, self.model, self.output_dim)
        global_net.load_state_dict(self.net.state_dict())
        if self.alg == "moon":
            train_moon(
                self.net,
                global_net,
                self.prev_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.temperature,
                self.device,
            )
        elif self.alg == "fedprox":
            train_fedprox(
                self.net,
                global_net,
                self.trainloader,
                self.num_epochs,
                self.learning_rate,
                self.mu,
                self.device,
            )
        if not os.path.exists(os.path.join(self.model_dir, str(self.net_id))):
            os.makedirs(os.path.join(self.model_dir, str(self.net_id)))
        torch.save(self.net.state_dict(), os.path.join(self.model_dir, str(self.net_id), "prev_net.pt"))
        return self.get_parameters({}), len(self.trainloader), {"is_straggler": False}

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Implement distributed evaluation for a given client."""
        self.set_parameters(parameters)
        # loss, accuracy = test(self.net, self.valloader, self.device)
        # skip evaluation in the client-side
        loss = 0.0
        accuracy = 0.0
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def gen_client_fn(
    trainloaders: List[DataLoader],
    testloaders: List[DataLoader],
    cfg: DictConfig,
) -> Callable[[str], FlowerClient]:
    """Generate the client function that creates the Flower Clients."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("device:", device)
        # net = init_net(cfg.dataset.name, cfg.model.name, cfg.model.output_dim)

        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        trainloader = trainloaders[int(cid)]
        testloader = testloaders[int(cid)]

        return FlowerClient(
            int(cid),
            cfg.dataset.name,
            cfg.model.name,
            cfg.model.output_dim,
            trainloader,
            testloader,
            device,
            cfg.num_epochs,
            cfg.learning_rate,
            cfg.mu,
            cfg.temperature,
            cfg.model.dir,
            cfg.alg,
        )

    return client_fn
