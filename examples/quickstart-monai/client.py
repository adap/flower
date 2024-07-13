import argparse
import warnings
from collections import OrderedDict

import torch
from data import load_data
from model import test, train
from monai.networks.nets.densenet import DenseNet121

import flwr as fl

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, device):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epoch_num=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return loss, len(self.testloader.dataset), {"accuracy": accuracy}


if __name__ == "__main__":
    total_partitions = 10
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--partition-id", type=int, choices=range(total_partitions), required=True
    )
    args = parser.parse_args()

    # Load model and data (simple CNN, CIFAR-10)
    trainloader, _, testloader, num_class = load_data(
        total_partitions, args.partition_id
    )
    net = DenseNet121(spatial_dims=2, in_channels=1, out_channels=num_class).to(DEVICE)

    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient(net, trainloader, testloader, DEVICE),
    )
