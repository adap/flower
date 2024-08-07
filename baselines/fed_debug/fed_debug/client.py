"""Define your client class and a function to construct such clients.

Please overwrite `flwr.client.NumPyClient` or `flwr.client.Client` and create a function
to instantiate your client.
"""

import gc

import flwr as fl
import torch

from fed_debug.models import train_neural_network


class CNNFlowerClient(fl.client.NumPyClient):
    """Flower client for training a CNN model."""

    def __init__(self, config):
        """Initialize the client with the given configuration."""
        self.config = config

    def fit(self, parameters, config):
        """Train the model on the local dataset."""
        nk_client_data_points = len(self.config["client_data_train"])
        model_dict = self.config["model_dict"]

        set_parameters(model_dict["model"], parameters=parameters)
        train_dict = train_neural_network(
            {
                "lr": config["lr"],
                "epochs": config["local_epochs"],
                "batch_size": config["batch_size"],
                "model_dict": model_dict,
                "train_data": self.config["client_data_train"],
                "device": self.config["device"],
            }
        )

        model_dict["model"] = model_dict["model"].cpu()
        parameters = get_parameters(model_dict["model"])
        del model_dict["model"]
        del model_dict
        client_train_dict = {"cid": self.config["cid"]} | train_dict
        gc.collect()
        return parameters, nk_client_data_points, client_train_dict


def get_parameters(model):
    """Return model parameters as a list of NumPy ndarrays."""
    model = model.cpu()
    return [val.cpu().detach().clone().numpy() for _, val in model.state_dict().items()]


def set_parameters(net, parameters):
    """Set model parameters from a list of NumPy ndarrays."""
    net = net.cpu()
    params_dict = zip(net.state_dict().keys(), parameters)
    new_state_dict = {k: torch.from_numpy(v) for k, v in params_dict}
    net.load_state_dict(new_state_dict, strict=True)
