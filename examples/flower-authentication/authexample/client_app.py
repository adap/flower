"""authexample: An authenticated Flower / PyTorch app."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from authexample.task import (
    Net,
    get_weights,
    load_data_from_disk,
    set_weights,
    test,
    train,
)


# Define Flower Client
class FlowerClient(NumPyClient):
    def __init__(self, trainloader, valloader, local_epochs, learning_rate):
        self.net = Net()
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.lr = learning_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.lr,
            self.device,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to get the path to the dataset the SuperNode running
    # this ClientApp has access to
    dataset_path = context.node_config["dataset-path"]

    # Read run_config to fetch hyperparameters relevant to this run
    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data_from_disk(dataset_path, batch_size)
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]

    # Return Client instance
    return FlowerClient(trainloader, valloader, local_epochs, learning_rate).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
