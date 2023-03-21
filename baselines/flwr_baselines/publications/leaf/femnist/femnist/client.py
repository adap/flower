from typing import List

import flwr as fl
import numpy as np
import torch
from model import Net, test, train
from torch.utils.data import DataLoader


def get_parameters(net: torch.nn.Module) -> List[np.ndarray]:
    """Get parameters from a PyTorch network."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    """Set parameters to a PyTorch network."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = dict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    """Flower client for training with train and validation loss and accuracy
    that enables having training time in epochs or in batches."""

    def __init__(
        self,
        net: torch.nn.Module,
        trainloader: DataLoader,
        valloader: DataLoader,
        testloader: DataLoader,
        device: torch.device,
        num_epochs: int,
        learning_rate: float,
        num_batches: int = None,
    ):
        """

        Parameters
        ----------
        net: torch.nn.Module
            PyTorch model
        trainloader, valloader, testloader: torch.utils.data.DataLoader
            dataloaders with images and labels
        device: torch.device
            denotes CPU or GPU training
        num_epochs: int
            training time for each client locally
        learning_rate: float
            learning rate used locally for model updates
        num_batches: int
            length of local training in batches (either this or num_epoch is used,
            if num_epoch is not None then num_epochs is used)
        """
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_batches = num_batches

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        """Fit locally training model."""
        set_parameters(self.net, parameters)
        train_loss, train_acc, val_loss, val_acc = train(
            self.net,
            self.trainloader,
            self.valloader,
            epochs=self.num_epochs,
            learning_rate=self.learning_rate,
            device=self.device,
            n_batches=self.num_batches,
        )
        return (
            get_parameters(self.net),
            len(self.trainloader),
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
        )

    def evaluate(self, parameters, config):
        """Evaluate locally training model."""
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


def create_client(
    cid: str,
    trainloaders: List[DataLoader],
    valloaders: List[DataLoader],
    testloaders: List[DataLoader],
    device: torch.device,
    num_epochs: int,
    learning_rate: float,
    num_classes: int = 62,
    num_batches: int = None,
):
    """Create client for the flower simulation."""
    net = Net(num_classes).to(device)

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    testloader = testloaders[int(cid)]

    return FlowerClient(
        net,
        trainloader,
        valloader,
        testloader,
        device,
        num_epochs,
        learning_rate,
        num_batches,
    )
