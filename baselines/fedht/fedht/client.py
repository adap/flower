from collections import OrderedDict
from typing import cast

import torch
from flwr.client import Client, NumPyClient
from flwr.common import Context
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedht.model import test, train
from fedht.utils import MyDataset


# SimII client
class SimIIClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        testloader,
        model,
        num_obs,
        num_features,
        num_classes,
        cfg: DictConfig,
    ) -> None:
        """SimII client for simulation II experimentation."""

        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.num_obs = num_obs
        self.num_features = num_features
        self.num_classes = num_classes
        self.cfg = cfg

    # get parameters from existing model
    def get_parameters(self, config):
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
        train(self.model, self.trainloader, self.cfg)

        return self.get_parameters(self.model), self.num_obs, {}

    def evaluate(self, parameters, config):
        """Evaluate model."""
        # set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # need to change from log_loss to torch.loss and change other metrics
        loss, accuracy = test(self.model, self.trainloader)

        return loss, self.num_obs, {"accuracy": accuracy}


# client fn for input into simulation
def generate_client_fn_simII(
    dataset, num_features, num_classes, model, cfg: DictConfig
):
    """Generates client function for simulated FL."""

    # def client_fn(cid: int):
    def client_fn(context: Context) -> Client:
        """Define client function for centralized metrics."""

        # Get node_config value to fetch partition_id
        partition_id = cast(int, context.node_config["partition-id"])

        # Load the partition data
        X_train, y_train = dataset
        num_obs = X_train.shape[1]
        test_dataset = train_dataset = MyDataset(
            X_train[int(partition_id), :, :], y_train[:, int(partition_id)]
        )
        trainloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

        return SimIIClient(
            trainloader, testloader, model, num_obs, num_features, num_classes, cfg
        ).to_client()

    return client_fn


# MNIST client
class MnistClient(NumPyClient):
    def __init__(
        self,
        trainloader,
        testloader,
        model,
        num_obs,
        num_features,
        num_classes,
        cfg: DictConfig,
    ) -> None:
        """MNIST client for MNIST experimentation."""

        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.num_obs = num_obs
        self.num_features = num_features
        self.num_classes = num_classes
        self.cfg = cfg

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
        train(self.model, self.trainloader, self.cfg)

        return self.get_parameters(self.model), self.num_obs, {}

    def evaluate(self, parameters, config):
        """Evaluate model."""

        # set model parameters
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # need to change from log_loss to torch.loss and change other metrics
        loss, accuracy = test(self.model, self.trainloader)

        return loss, self.num_obs, {"accuracy": accuracy}


# client fn for input into simulation
def generate_client_fn_mnist(
    dataset, num_features, num_classes, model, cfg: DictConfig
):
    """Generate client function for simulated FL."""

    # def client_fn(cid: int):
    def client_fn(context: Context) -> Client:
        """Define client function for centralized metrics."""

        # Get node_config value to fetch partition_id
        partition_id = cast(int, context.node_config["partition-id"])

        # Load the partition data
        train_dataset = dataset.load_partition(int(partition_id), "train").with_format(
            "numpy"
        )
        num_obs = train_dataset.num_rows
        test_dataset = dataset.load_partition(int(partition_id), "train").with_format(
            "numpy"
        )
        trainloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

        return MnistClient(
            trainloader, testloader, model, num_obs, num_features, num_classes, cfg
        ).to_client()

    return client_fn
