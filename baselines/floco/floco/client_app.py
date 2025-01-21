"""floco: A Flower Baseline."""
import os
import time

import torch

from flwr.common import (
    bytes_to_ndarray
)

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from floco.dataset import load_data
from floco.model import SimplexModel, get_weights, set_weights, test, train

from floco.server_app import DEVICE, load_datasets

class FlowerClient(NumPyClient):
    """A class defining the client."""

    def __init__(self, global_model, pers_model, trainloader, valloader, local_epochs):
        self.global_model = global_model
        self.pers_model = pers_model
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.global_model.subregion_parameters = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.global_model.to(self.device)

    def fit(self, parameters, config):
        """Train model using this client's data."""
        set_weights(self.global_model, parameters)
        self.global_model.training = True
        if all(key in config for key in ["center", "radius"]):
            self.global_model.subregion_parameters = (
                bytes_to_ndarray(config["center"]),
                bytes_to_ndarray(config["radius"])
            )
        train_loss = train(
            self.global_model,
            self.trainloader,
            self.local_epochs,
            self.device,
        )
        return (
            get_weights(self.global_model),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        """Evaluate model using this client's data."""
        set_weights(self.global_model, parameters)
        self.global_model.training = False
        loss, accuracy = test(self.global_model, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    # Load model and data
    global_model = SimplexModel(
        endpoints=context.run_config["endpoints"],
        seed=context.run_config["seed"],
    ).to(DEVICE)
    if context.run_config["pers-epoch"] > 0:
        pers_model = SimplexModel(
        endpoints=context.run_config["endpoints"],
        seed=context.run_config["seed"],
    ).to(DEVICE)
    else:
        pers_model = None
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader, _ = load_datasets(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(global_model, pers_model, trainloader, valloader, local_epochs).to_client()

# Flower ClientApp
app = ClientApp(client_fn)