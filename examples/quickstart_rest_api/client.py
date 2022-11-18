import flwr as fl


# Define Flower client
class DummyClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        print(parameters)
        return parameters, 42, {}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8000", client=DummyClient(), use_rest=True
)
