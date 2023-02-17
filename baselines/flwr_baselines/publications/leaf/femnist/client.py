from typing import List, Tuple, Callable

import torch
import numpy as np
import flwr as fl
from torch.utils.data import DataLoader

from model import train, test, Net


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = dict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(
            self,
            net: torch.nn.Module,
            trainloader: DataLoader,
            valloader: DataLoader,
            device: torch.device,
            num_epochs: int,
            learning_rate: float,
    ):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=self.num_epochs, learning_rate=self.learning_rate, device=self.device)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, device=self.device)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def full_client_fn(cid: str,
                   trainloaders: List[DataLoader],
                   testloaders: List[DataLoader],
                   device: torch.device,
                   # num_clients: int,
                   num_epochs: int,
                   learning_rate: float,
                   num_classes: int = 62,

                   ):
    # todo: remove redundant arguments (I assume to have already created the dataset instead of recreate it all the times
    net = Net(num_classes).to(device)

    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    testloader = testloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(
        net, trainloader, testloader, device, num_epochs, learning_rate
    )
