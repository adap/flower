"""$project_name: A Flower / HuggingFace Transformers app."""

import flwr as fl
from transformers import AutoModelForSequenceClassification

from $import_name.task import (
    get_weights,
    load_data,
    set_weights,
    train,
    test,
    CHECKPOINT,
    DEVICE,
)


# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def get_parameters(self, config):
        return get_weights(self.net)

    def set_parameters(self, parameters):
        set_weights(self.net, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1)
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": accuracy}


def client_fn(cid):
    # Load model and data
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)
    trainloader, valloader = load_data(int(cid), 2)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader).to_client()


# Flower ClientApp
app = fl.client.ClientApp(
    client_fn,
)
