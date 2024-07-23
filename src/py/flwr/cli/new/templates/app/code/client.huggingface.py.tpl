"""$project_name: A Flower / HuggingFace Transformers app."""

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
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
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs

    def get_parameters(self, config):
        return get_weights(self.net)

    def set_parameters(self, parameters):
        set_weights(self.net, parameters)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(
            self.net,
            self.trainloader,
            epochs=self.local_epochs,
        )
        return self.get_parameters(config={}), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader)
        return float(loss), len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = AutoModelForSequenceClassification.from_pretrained(
        CHECKPOINT, num_labels=2
    ).to(DEVICE)

    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
