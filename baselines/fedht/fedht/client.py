"""Generate client for fedht baseline."""

from collections import OrderedDict
from typing import cast

import torch
import copy
from flwr.client import Client, NumPyClient
from flwr.common import Context
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fedht.model import test, train
from fedht.utils import MyDataset
from fedht.model import LogisticRegression

# SimII client
class SimIIClient(NumPyClient):
    """Define SimIIClient class."""

    def __init__(
        self,
        trainloader,
        testloader,
        model,
        num_obs,
        cfg: DictConfig,
        device
    ) -> None:
        """SimII client for simulation II experimentation."""
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.num_obs = num_obs
        self.num_features = cfg.num_features
        self.num_classes = cfg.num_classes
        self.cfg = cfg
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


# client fn for input into simulation
def generate_client_fn_simII(
    dataset, cfg: DictConfig
):
    """Generate client function for simulated FL."""

    # def client_fn(cid: int):
    def client_fn(context: Context) -> Client:
        """Define client function for centralized metrics."""
        # Get node_config value to fetch partition_id
        partition_id = cast(int, context.node_config["partition-id"])

        # Load the partition data
        X_train, y_train = copy.deepcopy(dataset)
        num_obs = X_train.shape[1]
        test_dataset = train_dataset = MyDataset(
            X_train[int(partition_id), :, :], y_train[:, int(partition_id)]
        )
        trainloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=True)

        # define model and set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = LogisticRegression(cfg.num_features, cfg.num_classes).to(device)

        return SimIIClient(
            trainloader, testloader, model, num_obs, cfg, device
        ).to_client()

    return client_fn


# MNIST client
class MnistClient(NumPyClient):
    """Define MnistClient class."""

    def __init__(
        self,
        trainloader,
        testloader,
        model,
        num_obs,
        cfg: DictConfig,
        device
    ) -> None:
        """MNIST client for MNIST experimentation."""
        self.trainloader = trainloader
        self.testloader = testloader
        self.model = model
        self.num_obs = num_obs
        self.num_features = cfg.num_features
        self.num_classes = cfg.num_classes
        self.cfg = cfg
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


# client fn for input into simulation
def generate_client_fn_mnist(
    dataset, cfg: DictConfig
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

        # define model and set device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = LogisticRegression(cfg.num_features, cfg.num_classes).to(device)

        return MnistClient(
            trainloader, testloader, model, num_obs, cfg, device
        ).to_client()

    return client_fn
