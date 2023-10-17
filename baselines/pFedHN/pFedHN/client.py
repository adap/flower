"""Client Handling."""

from collections import OrderedDict

import flwr as fl
import torch

from pFedHN.models import CNNTarget
from pFedHN.trainer import train
from pFedHN.utils import get_device


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

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        cid,
        trainloader,
        testloader,
        valloader,
        cfg,
        local_layers,
        local_optims,
        local,
    ) -> None:
        super().__init__()

        self.cid = cid
        self.trainloader = trainloader
        self.testloader = testloader
        self.valloader = valloader
        self.local_layers = local_layers
        self.local_optims = local_optims
        self.local = local
        self.device = get_device()
        self.epochs = cfg.client.num_epochs
        self.n_kernels = cfg.model.n_kernels
        self.learning_rate = cfg.model.inner_lr
        self.weight_decay = cfg.model.wd
        self.net = CNNTarget(
            in_channels=cfg.model.in_channels,
            n_kernels=self.n_kernels,
            out_dim=cfg.model.out_dim,
            local=self.local,
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

        test_loss, test_acc, final_state = train(
            self.net,
            self.trainloader,
            self.testloader,
            self.valloader,
            self.local_layers,
            self.local_optims,
            self.local,
            self.epochs,
            self.learning_rate,
            self.weight_decay,
            self.device,
            self.cid,
        )

        # Calculating delta theta
        delta_theta = OrderedDict(
            {k: v - final_state[k] for k, v in inner_state.items()}
        )

        delta_theta = [val.cpu().numpy() for _, val in delta_theta.items()]

        return (
            delta_theta,
            len(self.trainloader),
            {"test_loss": test_loss, "test_acc": test_acc},
        )


# pylint: disable=too-many-arguments
def generate_client_fn(
    trainloaders,
    testloaders,
    valloaders,
    config,
    local_layers,
    local_optims,
    local=False,
):
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
        return FlowerClient(
            cid,
            trainloaders,
            testloaders,
            valloaders,
            config,
            local_layers,
            local_optims,
            local,
        )

    return client_fn
