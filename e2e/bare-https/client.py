import flwr as fl
import numpy as np
from pathlib import Path


model_params = np.array([1])
objective = 5

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model_params

    def fit(self, parameters, config):
        model_params = parameters
        model_params = [param * (objective/np.mean(param)) for param in model_params]
        return model_params, 1, {}

    def evaluate(self, parameters, config):
        model_params = parameters
        loss = min(np.abs(1 - np.mean(model_params)/objective), 1)
        accuracy = 1 - loss
        return loss, 1, {"accuracy": accuracy}

def client_fn(cid):
    return FlowerClient()


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080", 
        client=FlowerClient(),
        root_certificates=Path("certificates/ca.crt").read_bytes(),
    )
