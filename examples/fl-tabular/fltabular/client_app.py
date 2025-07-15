"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fltabular.task import (
    IncomeClassifier,
    evaluate,
    get_weights,
    load_data,
    set_weights,
    train,
)


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]

    train_loader, test_loader = load_data(
        partition_id=partition_id, num_partitions=context.node_config["num-partitions"]
    )
    net = IncomeClassifier()
    return FlowerClient(net, train_loader, test_loader).to_client()


app = ClientApp(client_fn=client_fn)
