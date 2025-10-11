# my_client.py
import flwr as fl
import numpy as np

# A Flower client needs to be a class that inherits from fl.client.NumPyClient
class SimpleClient(fl.client.NumPyClient):
    """A very basic Flower client."""

    # This function is called by the server to get the model parameters
    def get_parameters(self, config):
        print("Server asked for my parameters!")
        # We don't have a real model, so just send a dummy array
        return [np.zeros((2, 2))]

    # This is where the training would happen
    def fit(self, parameters, config):
        print("Server told me to train (fit)!")
        # We're not training, so just return the same dummy parameters
        return parameters, 10, {} # (parameters, num_examples, metrics)

    # This is where evaluation would happen
    def evaluate(self, parameters, config):
        print("Server told me to evaluate!")
        # Return a dummy loss and accuracy
        return 0.5, 10, {"accuracy": 0.95} # (loss, num_examples, metrics)

# Now, we start this client and tell it to connect to our server
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=SimpleClient(),
)