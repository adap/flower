"""Client Handling."""

from collections import OrderedDict

import flwr as fl
import torch

from pFedHN.models import CNNTarget
from pFedHN.trainer import test, train


# pylint: disable=too-many-instance-attributes
class FlowerClient(fl.client.NumPyClient):
    """Flower client for federated learning.

    Args:
        cid (str): The client ID.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.
        valloader (torch.utils.data.DataLoader): DataLoader for validation data
        cfg (Config): Hydra Configuration.
        local_layers (list): List of local layers.
        local_optims (list): List of local optimizers.
        local (bool): Whether to use local layers.

    Attributes
    ----------
        cid (str): The client ID.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        testloader (torch.utils.data.DataLoader): DataLoader for test data.
        valloader (torch.utils.data.DataLoader): DataLoader for validation data
        local_layers (list): List of local layers.
        local_optims (list): List of local optimizers.
        local (bool): Whether to use local layers.
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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

        """
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v).to(self.device)
                for k, v in zip(self.net.state_dict().keys(), parameters)
            }
        )
        self.net.to(self.device).load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Perform federated training on the client.

        Args:
            parameters (list): List of parameter values.
            config (dict): Configuration dictionary.

        Returns
        -------
            tuple: A tuple containing final_state(parameters),
                   the number of training samples, and metrics.
        """
        self.set_parameters(parameters)

        final_state = train(
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

        final_state_val = [val.cpu().numpy() for _, val in final_state.items()]

        return (
            final_state_val,
            len(self.trainloader),
            {},
        )

    def evaluate(self, parameters, config):
        """Evaluate function."""
        self.set_parameters(parameters)
        eval_loss, eval_correct, eval_total = test(
            self.net,
            self.testloader,
            self.local,
            self.local_layers,
            self.cid,
            self.device,
        )
        return (
            float(eval_loss),
            len(self.testloader),
            {"correct": eval_correct, "total": eval_total},
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
        local_layers (list): List of local layers.
        local_optims (list): List of local optimizers.
        local (bool): Whether to use local layers.


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
