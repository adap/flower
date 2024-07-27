import argparse
import warnings
from collections import OrderedDict

import torch
from monai.networks.nets.efficientnet import EfficientNetBN

from flwr.common import Context
from flwr.client import NumPyClient, ClientApp


from monaiexample.task import load_data
from monaiexample.task import test, train

warnings.filterwarnings("ignore", category=UserWarning)


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epoch_num=1, device=self.device)
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader), {"accuracy": accuracy}


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader = load_data(num_partitions, partition_id)
    net = EfficientNetBN(model_name="efficientnet-b0", in_channels=1, num_classes=6)

    return FlowerClient(net, trainloader, valloader).to_client()


app = ClientApp(client_fn=client_fn)
