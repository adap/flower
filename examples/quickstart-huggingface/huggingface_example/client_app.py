"""huggingface_example: A Flower / Hugging Face app."""

import warnings

import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from transformers import logging
from huggingface_example.task import (
    train,
    test,
    load_data,
    set_params,
    get_params,
    get_model,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# To mute warnings reminding that we need to train the model to a downstream task
# This is something this example does.
logging.set_verbosity_error()


# Flower client
class IMDBClient(NumPyClient):
    def __init__(self, model_name, trainloader, testloader) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = get_model(model_name)
        self.net.to(self.device)

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        set_params(self.net, parameters)
        train(self.net, self.trainloader, epochs=1, device=self.device)
        return get_params(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        set_params(self.net, parameters)
        loss, accuracy = test(self.net, self.testloader, device=self.device)
        return float(loss), len(self.testloader), {"accuracy": float(accuracy)}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    trainloader, testloader = load_data(partition_id, num_partitions, model_name)

    return IMDBClient(model_name, trainloader, testloader).to_client()


app = ClientApp(client_fn=client_fn)
