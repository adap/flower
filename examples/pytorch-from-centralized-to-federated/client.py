"""Flower client example using PyTorch for CIFAR-10 image classification."""

import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import cifar
import flwr as fl

disable_progress_bar()


USE_FEDBN: bool = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Flower Client
class CifarClient(fl.client.NumPyClient):
    """Flower client implementing CIFAR-10 image classification using PyTorch."""

    def __init__(
        self,
        model: cifar.Net,
        trainloader: DataLoader,
        testloader: DataLoader,
    ) -> None:
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        self.model.train()
        if USE_FEDBN:
            # Return model parameters as a list of NumPy ndarrays, excluding
            # parameters of BN layers when using FedBN
            return [
                val.cpu().numpy()
                for name, val in self.model.state_dict().items()
                if "bn" not in name
            ]
        else:
            # Return model parameters as a list of NumPy ndarrays
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        self.model.train()
        if USE_FEDBN:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)
        else:
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[List[np.ndarray], int, Dict]:
        # Set model parameters, train model, return updated model parameters
        self.set_parameters(parameters)
        cifar.train(self.model, self.trainloader, epochs=1, device=DEVICE)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        # Set model parameters, evaluate model on local test dataset, return result
        self.set_parameters(parameters)
        loss, accuracy = cifar.test(self.model, self.testloader, device=DEVICE)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def main() -> None:
    """Load data, start CifarClient."""
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition-id", type=int, required=True, choices=range(0, 10))
    args = parser.parse_args()

    # Load data
    trainloader, testloader = cifar.load_data(args.partition_id)

    # Load model
    model = cifar.Net().to(DEVICE).train()

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))["img"].to(DEVICE))

    # Start client
    client = CifarClient(model, trainloader, testloader).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
