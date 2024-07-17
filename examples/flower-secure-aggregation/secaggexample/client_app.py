import time

from flwr.common import Context
from flwr.client import ClientApp, NumPyClient, Client
from flwr.client.mod import secaggplus_mod
import numpy as np


# Define FlowerClient and client_fn
class FlowerClient(NumPyClient):
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


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp.

    You can use settings in `context.run_config` to parameterize the
    construction of your Client. You could use the `context.node_config` to
    , for example, indicate which dataset to load (e.g accesing the partition-id).
    """
    return FlowerClient().to_client()


# Flower ClientApp
app = ClientApp(
    client_fn=client_fn,
    mods=[
        secaggplus_mod,
    ],
)
