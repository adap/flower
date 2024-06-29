"""proto-example: A Flower / PyTorch app."""

from typing import Optional
from flwr.client import NumPyClient, ClientApp

from proto_example.task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(self.net, self.trainloader, self.valloader, 1, DEVICE)
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(node_id: int, partition_id: Optional[int]):
    # Load model and data
    net = Net().to(DEVICE)
    print(f"{node_id = } // {partition_id = }")
    trainloader, valloader = load_data(partition_id, 2)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
