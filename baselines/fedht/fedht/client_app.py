"""Generate client for fedht baseline."""

from collections import OrderedDict
from typing import cast

import torch
import copy
import numpy as np
from flwr.client import Client, NumPyClient, ClientApp
from flwr.common import Context
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedht.model import test, train
from fedht.utils import MyDataset, load_data
from fedht.model import LogisticRegression
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import PathologicalPartitioner

# MNIST client
class MnistClient(NumPyClient):
    """Define MnistClient class."""

    def __init__(
        self,
        trainloader,
        testloader,
        model,
        num_obs,
        context,
        device
    ) -> None:
        """MNIST client for MNIST experimentation."""
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.num_obs = num_obs
        self.num_features = context.run_config["num_features"]
        self.num_classes = context.run_config["num_classes"]
        self.cfg = context
        self.device = device

    # get parameters from existing model
    def get_parameters(self, config):
        """Get parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Fit model."""
        # set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # train model
        self.model.train()

        # training for local epochs defined by config
        train(self.model, self.trainloader, self.cfg, self.device)

        return self.get_parameters(self.model), self.num_obs, {}

    def evaluate(self, parameters, config):
        """Evaluate model."""
        # set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # need to change from log_loss to torch.loss and change other metrics
        loss, accuracy = test(self.model, self.trainloader, self.device)

        return loss, self.num_obs, {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    """Define client function for centralized metrics."""

    num_features = context.run_config["num_features"]
    num_classes = context.run_config["num_classes"]
    
    trainloader, testloader, num_obs = load_data(context)

    # define model and set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LogisticRegression(num_features, num_classes).to(device)

    return MnistClient(
        trainloader, testloader, model, num_obs, context, device
    ).to_client()

app = ClientApp(
    client_fn=client_fn,
)
