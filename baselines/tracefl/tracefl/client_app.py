"""TraceFL Client Application Module.

This module implements the client-side functionality for the TraceFL federated learning
system. It provides the client implementation that handles local model training and
evaluation.
"""

import copy
import json

import toml
import torch
from omegaconf import OmegaConf

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from tracefl.dataset import (
    get_clients_server_data,
)
from tracefl.fls import FLSimulation
from tracefl.models import (
    get_parameters,
    initialize_model,
    set_parameters,
    train_neural_network,
)


class FlowerClient(NumPyClient):
    """A Flower client implementation for TraceFL.

    This client handles local model training and evaluation using the provided neural
    network model and data loaders. It supports standard federated learning operations
    like getting/setting weights and performing local training rounds.
    """

    def __init__(
        self,
        net,
        trainloader,
        local_epochs,
        partition_id,
        cfg,
        ds_dict,
        arch,
        model_dict,
        client2data,
    ):
        """Initialize the TraceFL client.

        Args:
            net: The neural network model to train
            trainloader: DataLoader for training data
            local_epochs: Number of local training epochs
            partition_id: Client's partition ID
            cfg: Configuration object
            ds_dict: Dictionary containing dataset information
            arch: Model architecture
            model_dict: Dictionary containing model information
            client2data: Mapping of client IDs to their data
        """
        self.net = net
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.cid = partition_id
        self.is_faulty = False
        self.cfg = cfg
        self.ds_dict = ds_dict
        self.arch = arch
        self.model_dict = model_dict
        self.client2data = client2data

    def fit(self, parameters, config):
        """Train the model using the client's local data.

        Args:
            parameters: Current model parameters
            config: Configuration dictionary containing training parameters

        Returns
        -------
            tuple: Updated model parameters, number of training examples, and metrics
        """
        set_parameters(self.net, parameters)

        tconfig = {
            "arch": self.arch,
            "model_dict": self.model_dict,
            "train_data": self.ds_dict["client2data"][str(self.cid)],
            "batch_size": self.cfg.tool.tracefl.client.batch_size,
            "lr": self.cfg.tool.tracefl.client.lr,
            "epochs": self.cfg.tool.tracefl.client.epochs,
            "device": self.device,
        }
        train_result = train_neural_network(tconfig)
        label_counts = {}
        labels = self.ds_dict["client2data"][str(self.cid)]["label"]
        for label in labels:
            label = int(label)
            label_counts[label] = label_counts.get(label, 0) + 1

        return (
            get_parameters(self.net),
            len(self.ds_dict["client2data"][str(self.cid)]),
            {
                "train_loss": train_result["train_loss"],
                "train_accuracy": train_result["train_accuracy"],
                "client_message": "Hello from Client!",
                "cid": self.cid,
                "class_distribution": json.dumps(label_counts),
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate the model on the client's local test data.

        Args:
            parameters: Current model parameters
            config: Configuration dictionary containing evaluation parameters

        Returns
        -------
            tuple: Loss value, number of test examples, and evaluation metrics
        """
        pass


def client_fn(context: Context):
    """Create and configure a TraceFL client instance.

    Args:
        context: Flower context containing configuration and state

    Returns
    -------
        Client: Configured TraceFL client instance
    """
    possible_configs = ["exp_1", "exp_2"]
    config_key = next((k for k in possible_configs if k in context.run_config), "exp_1")
    config_path = str(context.run_config[config_key])
    config = toml.load(config_path)
    cfg = OmegaConf.create(config)

    partition_id = int(context.node_config["partition-id"])
    ds_dict = get_clients_server_data(cfg)

    client_train_data = ds_dict["client2data"].get(str(partition_id))

    sim = FLSimulation(copy.deepcopy(cfg), 0.0, 0.0, 0.0, 0.0)
    sim.set_clients_data(ds_dict["client2data"])
    sim.set_strategy()
    model_dict = initialize_model(cfg.tool.tracefl.model.name, cfg.tool.tracefl.dataset)
    local_epochs = int(context.run_config["local-epochs"])
    return FlowerClient(
        model_dict["model"],
        client_train_data,
        local_epochs,
        partition_id,
        cfg,
        ds_dict,
        cfg.tool.tracefl.model.arch,
        model_dict,
        ds_dict["client2data"],
    ).to_client()


app = ClientApp(client_fn=client_fn)
