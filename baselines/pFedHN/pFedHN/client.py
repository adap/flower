"""Client Handling."""

from collections import OrderedDict

import flwr as fl
import torch

from pFedHN.models import CNNTarget
from pFedHN.trainer import train


# pylint: disable=too-many-instance-attributes
class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning.

    Args:
        cid (str): The client ID.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.
        cfg (Config): Hydra Configuration.

    Attributes
    ----------
        cid (str): The client ID.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): The device to run the model on.
        epochs (int): Number of training epochs.
        n_kernels (int): Number of convolutional kernels.
        learning_rate (float): Learning rate.
        weight_decay (float): Weight decay.
        net (CNNTarget): Target neural network model.
    """

    def __init__(self, cid, trainloader, testloader, cfg) -> None:
        super().__init__()

        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = torch.device(cfg.model.device)
        self.epochs = cfg.client.num_epochs
        self.n_kernels = cfg.model.n_kernels
        self.learning_rate = cfg.model.inner_lr
        self.weight_decay = cfg.model.wd
        self.net = CNNTarget(
            in_channels=cfg.model.in_channels,
            n_kernels=self.n_kernels,
            out_dim=cfg.model.out_dim,
        )

    def set_parameters(self, parameters):
        """Set the target network parameters using the parameters from the server.

        Args:
            parameters (list): List of parameter values.

        Returns
        -------
            OrderedDict: The inner_state_dict of the target network.
        """
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v)
                for k, v in zip(self.net.state_dict().keys(), parameters)
            }
        )
        self.net.load_state_dict(state_dict, strict=True)
        return state_dict

    def get_parameters(self, config):
        """Get the target network parameters and send them to the server.

        Args:
            config (Parameter): Delta Theta.

        Returns
        -------
            list: List of parameter values.
        """
        return [val.cpu().numpy() for _, val in config.items()]

    def fit(self, parameters, config):
        """Perform federated training on the client.

        Args:
            parameters (list): List of parameter values.
            config (dict): Configuration dictionary.

        Returns
        -------
            tuple: A tuple containing delta theta (parameter updates),
                   the number of training samples, and metrics.
        """
        inner_state = self.set_parameters(parameters)

        train(
            self.net,
            self.trainloader,
            self.testloader,
            self.epochs,
            self.learning_rate,
            self.weight_decay,
            self.device,
            self.cid,
        )

        final_state = self.net.state_dict()

        # Calculating delta theta
        delta_theta = OrderedDict(
            {k: v - final_state[k] for k, v in inner_state.items()}
        )

        return self.get_parameters(delta_theta), len(self.trainloader), {}


def generate_client_fn(trainloaders, testloaders, config):
    """Generate a function which returns a new FlowerClient.

    Args:
        trainloaders (list): List of DataLoader objects for training data.
        testloaders (list): List of DataLoader objects for test data.
        config (Config): Hydra Configuration.

    Returns
    -------
        function: A function that creates a new FlowerClient instance.
    """

    def client_fn(cid: str):
        return FlowerClient(cid, trainloaders, testloaders, config)

    return client_fn
