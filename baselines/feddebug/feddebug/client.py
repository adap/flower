"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import gc

import flwr as fl
import torch
from flwr.common.logger import log
from logging import DEBUG, INFO
from feddebug.models import initialize_model, train_neural_network
from feddebug.utils import get_parameters, set_parameters


class CNNFlowerClient(fl.client.NumPyClient):
    """Flower client for training a CNN model."""

    def __init__(self, args):
        """Initialize the client with the given configuration."""
        self.args = args

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        nk_client_data_points = len(self.args["client_data_train"])
        model_dict = self.args["model_dict"]

        set_parameters(model_dict["model"], parameters=parameters)
        train_dict = train_neural_network(
            {
                "lr": config["lr"],
                "epochs": config["local_epochs"],
                "batch_size": config["batch_size"],
                "model_dict": model_dict,
                "train_data": self.args["client_data_train"],
                "device": self.args["device"],
            }
        )

        model_dict["model"] = model_dict["model"].cpu()
        parameters = get_parameters(model_dict["model"])
        del model_dict["model"]
        del model_dict
        client_train_dict = {"cid": self.args["cid"]} | train_dict
        gc.collect()
        log(INFO, f"Client {self.args['cid']} trained.")
        return parameters, nk_client_data_points, client_train_dict


def gen_client_func(cfg, client2data, cid):
    """Give the new client."""
    model_dict = initialize_model(cfg.model.name, cfg.dataset)
    args = {
        "cid": cid,
        "model_dict": model_dict,
        "client_data_train": client2data[cid],
        "device": torch.device(cfg.device),
    }
    client = CNNFlowerClient(args).to_client()
    return client
