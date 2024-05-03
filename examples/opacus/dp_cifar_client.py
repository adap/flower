import math

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import flwr as fl

from dp_cifar_main import Net, DPCifarClient, PARAMS

# Setup for running a single client manually (alternatively use simulation code in 'dp_cifar_simulation').


def load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    data = CIFAR10("./data", train=True, download=True, transform=transform)
    split = math.floor(len(data) * PARAMS["train_split"])
    trainset = torch.utils.data.Subset(data, list(range(0, split)))
    trainset = torch.utils.data.Subset(data, list(range(split, len(data))))
    trainloader = DataLoader(trainset, PARAMS["batch_size"])
    testloader = DataLoader(trainset, PARAMS["batch_size"])
    sample_rate = PARAMS["batch_size"] / len(trainset)
    return trainloader, testloader, sample_rate


model = Net()
trainloader, testloader, sample_rate = load_data()
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=DPCifarClient(model, trainloader, testloader).to_client(),
)
