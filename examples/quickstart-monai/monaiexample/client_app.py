"""monaiexample: A Flower / MONAI app."""

import torch
from flwr.common import Context
from flwr.client import NumPyClient, ClientApp

from monaiexample.task import load_data, load_model, test, train, get_params, set_params


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        set_params(self.net, parameters)
        train(self.net, self.trainloader, epoch_num=1, device=self.device)
        return get_params(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_params(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)
        return loss, len(self.valloader), {"accuracy": accuracy}


def client_fn(context: Context):

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    batch_size = context.run_config["batch-size"]
    trainloader, valloader = load_data(num_partitions, partition_id, batch_size)
    net = load_model()

    return FlowerClient(net, trainloader, valloader).to_client()


app = ClientApp(client_fn=client_fn)
