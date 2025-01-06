"""fastai_example: A Flower / Fastai app."""

import warnings
from typing import Any

from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.all import error_rate, squeezenet1_1
from fastai.vision.data import DataLoaders
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context

from fastai_example.task import get_params, load_data, set_params

warnings.filterwarnings("ignore", category=UserWarning)


# Define Flower client
class FlowerClient(NumPyClient):
    def __init__(self, learn, dls) -> None:
        self.learn = learn
        self.dls = dls

    def fit(self, parameters, config) -> tuple[list, int, dict]:
        set_params(self.learn.model, parameters)
        with self.learn.no_bar(), self.learn.no_logging():
            self.learn.fit(1)
        return get_params(self.learn.model), len(self.dls.train), {}

    def evaluate(self, parameters, config) -> tuple[Any, int, dict[str, Any]]:
        set_params(self.learn.model, parameters)
        with self.learn.no_bar(), self.learn.no_logging():
            loss, error_rate = self.learn.validate()
        return loss, len(self.dls.valid), {"accuracy": 1 - error_rate}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    trainloader, valloader, _ = load_data(partition_id, num_partitions)
    dls = DataLoaders(trainloader, valloader)
    model = squeezenet1_1()
    learn = Learner(
        dls,
        model,
        loss_func=CrossEntropyLossFlat(),
        metrics=error_rate,
    )
    return FlowerClient(learn, dls).to_client()


app = ClientApp(client_fn=client_fn)
