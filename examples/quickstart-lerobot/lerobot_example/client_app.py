"""huggingface_example: A Flower / Hugging Face app."""

import warnings

import torch
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from transformers import logging
from logging import INFO

from flwr.common.logger import log
from lerobot_example.task import (
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


# Check if GPU is available
if torch.cuda.is_available():
    nn_device = torch.device("cuda")
    print("GPU is available. Device set to:", nn_device)
else:
    nn_device = torch.device("cpu")
    print(f"GPU is not available. Device set to: {nn_device}. Inference will be slower than on GPU.")


# Flower client
class LeRobotClient(NumPyClient):
    def __init__(self, partition_id, model_name, local_epochs, trainloader, testloader) -> None:
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.testloader = testloader
        self.net = get_model(model_name=model_name, dataset=trainloader.dataset)
        self.local_epochs = local_epochs
        policy = self.net
        self.device = nn_device
        if self.device == torch.device("cpu"):
            # Decrease the number of reverse-diffusion steps (trades off a bit of quality for 10x speed)
            policy.diffusion.num_inference_steps = 10        
        policy.to(self.device)

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        set_params(self.net, parameters)
        train(partition_id=self.partition_id, net=self.net, trainloader=self.trainloader, epochs=self.local_epochs, device=self.device)
        return get_params(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config) -> tuple[float, int, dict[str, float]]:
        set_params(self.net, parameters)
        # loss, accuracy = test(partition_id=self.partition_id, net=self.net, device=self.device)
        # return float(loss), len(self.testloader), {"accuracy": float(accuracy)}
        loss, accuracy= test(partition_id=self.partition_id, net=self.net, device=self.device)
        testset_len = 1 # we test on one gym generated task
        return float(loss), testset_len, {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    log(INFO, f"partition_id={partition_id}, num_partitions={num_partitions}")

    # Read the run config to get settings to configure the Client
    model_name = context.run_config["model-name"]
    local_epochs = int(context.run_config["local-epochs"])
    log(INFO, f"local_epochs={local_epochs}")
    trainloader, testloader = load_data(partition_id, num_partitions, model_name, device=nn_device)

    return LeRobotClient(partition_id=partition_id, model_name=model_name, local_epochs=local_epochs, trainloader=trainloader, testloader=testloader).to_client()


app = ClientApp(client_fn=client_fn)
