"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

import pytorch_lightning as pl
from datasets.utils.logging import disable_progress_bar
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

disable_progress_bar()

from pytorchlightning_example.task import (
    LitAutoEncoder,
    get_parameters,
    load_data,
    set_parameters,
)


class FlowerClient(NumPyClient):
    def __init__(self, train_loader, val_loader, test_loader, max_epochs):
        self.model = LitAutoEncoder()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_parameters(self.model, parameters)

        trainer = pl.Trainer(max_epochs=self.max_epochs, enable_progress_bar=False)
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_parameters(self.model, parameters)

        trainer = pl.Trainer(enable_progress_bar=False)
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, len(self.test_loader.dataset), {}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader, test_loader = load_data(partition_id, num_partitions)

    # Read run_config to fetch hyperparameters relevant to this run
    max_epochs = context.run_config["max-epochs"]
    return FlowerClient(train_loader, val_loader, test_loader, max_epochs).to_client()


app = ClientApp(client_fn=client_fn)
