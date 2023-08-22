import copy
from collections import OrderedDict
from typing import List, Callable, Tuple, Dict

import flwr as fl
import torch
from flwr.common import Scalar, NDArrays
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

from fedexp.models import train
from fedexp.utils import get_parameters


class FlowerClient(fl.client.NumPyClient):
    def __init__(self,
                 cid: int,
                 net: torch.nn.Module,
                 trainloader: DataLoader,
                 device: torch.device,
                 num_epochs: int,
                 p,
                 ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.device = device
        self.num_epochs = num_epochs
        self.p = p
        print(f"Client {self.cid} created.")

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Returns the parameters of the current net."""
        return get_parameters(self.net)

    def set_parameters(self, parameters: NDArrays) -> None:
        """Changes the parameters of the model using the given ones."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Implements distributed fit function for a given client."""
        self.set_parameters(parameters)
        print(f"Client {self.cid} Training...")
        prev_net = copy.deepcopy(self.net)
        train(
            self.net,
            self.trainloader,
            self.device,
            epochs=self.num_epochs,
            hyperparams=config,
        )
        with torch.no_grad():
            vec_curr = parameters_to_vector(self.net.parameters())
            vec_prev = parameters_to_vector(prev_net.parameters())
            params_delta_vec = vec_curr - vec_prev
            grad = params_delta_vec
            grad_p = self.p * grad
            grad_norm = self.p * torch.linalg.norm(grad) ** 2
        return self.get_parameters({}), len(self.trainloader), {"p": self.p,
                                                                "grad_p": grad_p,
                                                                "grad_norm": grad_norm,
                                                                "epsilon": config["epsilon"]
                                                                }


def gen_client_fn(trainloaders: List[DataLoader],
                  model: DictConfig,
                  num_epochs: int,
                  args: Dict,
                  ) -> Callable[[str], FlowerClient]:
    """Return a function which creates a new FlowerClient for a given cid."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a new FlowerClient for a given cid."""

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net = instantiate(model).to(device)
        trainloader = trainloaders[int(cid)]
        p = args["p"][int(cid)]

        return FlowerClient(
            cid=int(cid),
            net=net,
            trainloader=trainloader,
            device=device,
            num_epochs=num_epochs,
            p=p,
        )

    return client_fn
