import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

model_params = np.array([1])
objective = 5


# Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return model_params

    def fit(self, parameters, config):
        model_params = parameters
        model_params = [param * (objective / np.mean(param)) for param in model_params]
        return model_params, 1, {}

    def evaluate(self, parameters, config):
        model_params = parameters
        loss = min(np.abs(1 - np.mean(model_params) / objective), 1)
        accuracy = 1 - loss
        return loss, 1, {"accuracy": accuracy}


def client_fn(context: Context):
    return FlowerClient().to_client()


app = ClientApp(
    client_fn=client_fn,
)
