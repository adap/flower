import time

import numpy as np

import flwr as fl
from flwr.client.secure_aggregation import SecAggPlusHandler


# Define Flower client with the SecAgg+ protocol
class FlowerClient(fl.client.NumPyClient, SecAggPlusHandler):
    def fit(self, parameters, config):
        # Force a significant delay for teshing purposes
        if self._shared_state.sid == 0:
            print(f"Client {self._shared_state.sid} dropped for testing purposes.")
            time.sleep(4)
            return [np.ones(3)], 1, {}
        ret = [np.ones(3)]
        print(f"Client {self._shared_state.sid} uploading {ret[0]}")
        return ret, 1, {}


# Start Flower client
fl.client.start_client(
    server_address="0.0.0.0:9092",
    client=FlowerClient(),
    transport="grpc-rere",
)
