import flwr as fl
import mnist
import pytorch_lightning as pl
from collections import OrderedDict
import wandb

import torch


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self):
        encoder_params = _get_parameters(self.model.encoder)
        decoder_params = _get_parameters(self.model.decoder)
        return encoder_params + decoder_params

    def set_parameters(self, parameters):
        _set_parameters(self.model.encoder, parameters[:4])
        _set_parameters(self.model.decoder, parameters[4:])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=0)
        trainer.fit(self.model, self.train_loader, self.val_loader)
        results = trainer.validate(self.model, self.test_loader)
        val_loss = results[0]["val_loss"]
        wandb.log({"val_loss": val_loss})
        return self.get_parameters(), 55000, {}

    # As we are doing centralized evaluation we can ignore this
    def evaluate(self, parameters, config):
        pass


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def main() -> None:
    # Configure wandb
    config = wandb.config
    config.dataset = "mnist"
    config.model = "LitAutoEncoder"

    # Init wandb
    run = wandb.init(
        entity="flwr",
        project="quickstart_pytorch_lightning",
        name="client_" + wandb.util.generate_id(),
        config=config,
    )

    # Model and data
    model = mnist.LitAutoEncoder()
    train_loader, val_loader, test_loader = mnist.load_data()

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader)
    fl.client.start_numpy_client("[::]:8080", client)

    # Tell wandb that we are done
    run.finish()


if __name__ == "__main__":
    main()
