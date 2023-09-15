from collections import OrderedDict
from typing import Dict, Tuple

import flwr as fl
import torch
from flwr.common import NDArrays, Scalar

from pFedHN.models import CNNTarget
from pFedHN.trainer import train


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, trainloader, testloader, cfg) -> None:
        super().__init__()

        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device(cfg.model.device)
        self.epochs = cfg.client.num_epochs
        self.n_kernels = cfg.model.n_kernels
        self.lr = cfg.model.inner_lr
        self.wd = cfg.model.wd
        self.net = CNNTarget(
            in_channels=cfg.model.in_channels,
            n_kernels=self.n_kernels,
            out_dim=cfg.model.out_dim,
        )

    def set_parameters(self, parameters):
        """Setting the target network parameters using the parameters from the
        server."""

        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
        return state_dict

    def get_parameters(self, config):
        """Getting the target network parameters and sending them to the
        server."""

        return [val.cpu().numpy() for _, val in config.items()]

    def fit(self, parameters, config):
        inner_state = self.set_parameters(parameters)

        train(
            self.net,
            self.trainloader,
            self.testloader,
            self.epochs,
            self.lr,
            self.wd,
            self.device,
            self.cid,
        )

        final_state = self.net.state_dict()

        # calculating delta theta
        delta_theta = OrderedDict(
            {k: inner_state[k] - final_state[k] for k in inner_state.keys()}
        )

        return self.get_parameters(delta_theta), len(self.trainloader), {}


def generate_client_fn(trainloaders, testloaders, config):
    """Generate a function which returns a new FlowerClient."""

    def client_fn(cid: str):
        return FlowerClient(cid, trainloaders, testloaders, config)

    return client_fn
