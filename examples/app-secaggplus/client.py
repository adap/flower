import time

import numpy as np

import flwr as fl
from flwr.client.mod import secaggplus_mod


# Define the Flower client
class FlowerClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        # Instead of training and returning model parameters,
        # the client directly returns [1.0, 1.0, 1.0] for demonstration purposes.
        ret_vec = [np.ones(3)]
        # Force a significant delay for testing purposes
        if "drop" in config and config["drop"]:
            print(f"Client dropped for testing purposes.")
            time.sleep(8)
        else:
            print(f"Client uploading {ret_vec[0]}...")
        return ret_vec, 1, {}


def client_fn(cid: str):
    """."""
    return FlowerClient().to_client()


# To run this: `flower-client-app client:app`
app = fl.client.ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
