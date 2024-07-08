from collections import OrderedDict
from typing import Optional

import mnist
import pytorch_lightning as pl
import torch

import flwr as fl


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        encoder_params = _get_parameters(self.model.encoder)
        decoder_params = _get_parameters(self.model.decoder)
        return encoder_params + decoder_params

    def set_parameters(self, parameters):
        _set_parameters(self.model.encoder, parameters[:4])
        _set_parameters(self.model.decoder, parameters[4:])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return self.get_parameters(config={}), 55000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, 10000, {"loss": loss}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def client_fn(node_id: int, partition_id: Optional[int]):
    model = mnist.LitAutoEncoder()
    train_loader, val_loader, test_loader = mnist.load_data()

    # Flower client
    return FlowerClient(model, train_loader, val_loader, test_loader).to_client()


app = fl.client.ClientApp(
    client_fn=client_fn,
)


def main() -> None:
    # Model and data
    model = mnist.LitAutoEncoder()
    train_loader, val_loader, test_loader = mnist.load_data()

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
