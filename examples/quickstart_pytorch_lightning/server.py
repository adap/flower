import flwr as fl
import pytorch_lightning as pl
import torch
from collections import OrderedDict
import wandb

import mnist


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn():
    """Return an evaluation function for server-side evaluation."""
    # Model and data
    model = mnist.LitAutoEncoder()
    _, val_loader, _ = mnist.load_data()
    trainer = pl.Trainer(progress_bar_refresh_rate=0)

    # The `evaluate` function will be called after every round
    def evaluate(parameters):
        _set_parameters(model.encoder, parameters[:4])
        _set_parameters(model.decoder, parameters[4:])

        # results will be similar to:
        # [{ 'val_loss': 0.14123 }]
        results = trainer.validate(model, val_loader)
        val_loss = results[0]["val_loss"]
        wandb.log({"val_loss": val_loss})
        return val_loss, {"val_loss": val_loss}

    return evaluate


def main() -> None:
    # Init wandb
    run = wandb.init(
        entity="flwr",
        project="quickstart_pytorch_lightning",
        name="server",
    )

    # Configure wandb
    config = wandb.config
    config.dataset = "mnist"
    config.model = "LitAutoEncoder"
    config.fraction_fit = 0.5
    config.fraction_eval = 0.5
    config.num_rounds = 10

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=config.fraction_fit,
        fraction_eval=config.fraction_eval,
        eval_fn=get_eval_fn(),
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="[::]:8080",
        config={"num_rounds": config.num_rounds},
        strategy=strategy,
    )

    # Tell wandb that we are done
    run.finish()


if __name__ == "__main__":
    main()
