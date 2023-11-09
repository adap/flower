import warnings
from collections import OrderedDict
from flwr.common import (
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    bytes_to_ndarray,
)

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from models import AlexNet, Generator
import argparse
from utils_pacs import make_dataloaders, train, test

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cpu")
num_classes = 7

parser = argparse.ArgumentParser()
parser.add_argument("--client_idx", type=int, required=True, help="Client index")
args = parser.parse_args()


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, MNIST)
net1 = AlexNet(num_classes=num_classes, latent_dim=4096, other_dim=1000).to(DEVICE)
trainloaders, testloaders, valloader = make_dataloaders(batch_size=64)
trainloader = trainloaders[args.client_idx]
testloader = testloaders[args.client_idx]


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self) -> None:
        super().__init__()
        self.model_len = len(net1.state_dict())

    def get_parameters(self, config):
        n1 = [val.cpu().numpy() for _, val in net1.state_dict().items()]
        return n1

    def set_parameters(self, parameters):
        params_dict1 = zip(net1.state_dict().keys(), parameters)
        state_dict1 = OrderedDict({k: torch.tensor(v) for k, v in params_dict1})

        net1.load_state_dict(state_dict1, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        # print(parameters_to_ndarrays(config["dict"])[0].shape())
        train(net1, trainloader, config, epochs=5)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net1, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)
