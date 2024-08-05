"""pytorchlightning_example: A Flower / PyTorch Lightning app.

Adapted from the PyTorch Lightning quickstart example.

Source: pytorchlightning.ai (2021/02/04)
"""

import pytorch_lightning as pl
from datasets.utils.logging import disable_progress_bar
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

disable_progress_bar()

from pytorchlightning_example.task import (
    LitAutoEncoder,
    get_parameters,
    set_parameters,
    load_data,
)


class FlowerClient(NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        encoder_params = get_parameters(self.model.encoder)
        decoder_params = get_parameters(self.model.decoder)
        return encoder_params + decoder_params

    def set_parameters(self, parameters):
        set_parameters(self.model.encoder, parameters[:4])
        set_parameters(self.model.decoder, parameters[4:])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return self.get_parameters(config={}), 55000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(enable_progress_bar=False)
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, 10000, {"loss": loss}


model = LitAutoEncoder()


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader, test_loader = load_data(partition_id, num_partitions)

    return FlowerClient(model, train_loader, val_loader, test_loader).to_client()


app = ClientApp(client_fn=client_fn)
