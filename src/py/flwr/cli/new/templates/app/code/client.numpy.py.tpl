"""$project_name: A Flower / $framework_str app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import numpy as np

from $import_name.task import get_dummy_model


class FlowerClient(NumPyClient):

    def fit(self, parameters, config):
        model = get_dummy_model()
        return ([model], 1, {})

    def evaluate(self, parameters, config):
        return float(0.0), 1, {"accuracy": float(1.0)}


def client_fn(context: Context):
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(client_fn=client_fn)
