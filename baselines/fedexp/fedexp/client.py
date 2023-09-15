"""Client for FedExp."""

import copy
from collections import OrderedDict
from typing import Callable, Dict, List, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader

from fedexp.models import train


class FlowerClient(fl.client.NumPyClient):
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        cid: int,
        net: torch.nn.Module,
        train_loader: DataLoader,
        device: str,
        num_epochs: int,
        data_ratio,
    ):  # pylint: disable=too-many-arguments
        print(f"Initializing Client {cid}")
        self.cid = cid
        self.net = net
        self.train_loader = train_loader
        self.device = torch.device(device)
        self.num_epochs = num_epochs
        self.data_ratio = data_ratio

    def _set_parameters(self, parameters: NDArrays) -> None:
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the network on the training set."""
        self._set_parameters(parameters)
        print(f"Client {self.cid} Training...")
        prev_net = copy.deepcopy(self.net)

        train(
            self.net,
            self.train_loader,
            self.device,
            epochs=self.num_epochs,
            hyperparams=config,
        )

        with torch.no_grad():
            vec_curr = parameters_to_vector(self.net.parameters())
            vec_prev = parameters_to_vector(prev_net.parameters())
            grad = vec_curr - vec_prev

        return (
            [],
            len(self.train_loader),
            {"grad": grad.to("cpu")},
        )


def gen_client_fn(
    train_loaders: List[DataLoader],
    model: DictConfig,
    num_epochs: int,
    args: Dict,
) -> Callable[[str], FlowerClient]:
    """Return a function which creates a new FlowerClient for a given cid."""

    def client_fn(cid: str) -> FlowerClient:
        """Create a new FlowerClient for a given cid."""

        return FlowerClient(
            cid=int(cid),
            net=instantiate(model).to(args["device"]),
            train_loader=train_loaders[int(cid)],
            device=args["device"],
            num_epochs=num_epochs,
            data_ratio=args["data_ratio"][int(cid)],
        )

    return client_fn
