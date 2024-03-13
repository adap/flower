"""$project_name: A Flower / PyTorch app."""

from flwr.client import NumPyClient, ClientApp
from flwr.cli.flower_toml import load_and_validate_with_defaults

from $project_name.task import (
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
    def __init__(self, net, trainloader, valloader) -> None:
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


# Load config
cfg, *_ = load_and_validate_with_defaults()

def client_fn(cid: str):
    # Load model and data
    net = Net().to(DEVICE)
    engine = cfg["flower"]["engine"]
    num_partitions = 2
    if "simulation" in engine:
        num_partitions = engine["simulation"]["supernode"]["num"]
    trainloader, valloader = load_data(int(cid), num_partitions)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader).to_client()


# Flower ClientApp
app = ClientApp(client_fn)
