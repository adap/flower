from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn


class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc = nn.Linear(5, 10)  # Simple Linear layer

    def forward(self, x):
        return self.fc(x)


def load_data():
    # Create some dummy data
    X = torch.randn(100, 5)  # 100 samples, 5 features
    y = (torch.sum(X, dim=1) > 0).float()  # Sum of features > 0 as positive label

    # Split the data
    X_train = X[:80]
    y_train = y[:80]
    X_test = X[80:]
    y_test = y[80:]
    return (X_train, y_train), (X_test, y_test)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data):
        super(FlowerClient, self).__init__()
        self.data = data
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        pass

    def fit(self, parameters, config):
        self.embedding = self.model(self.data[0][0])
        return self.embedding.detach().numpy(), 1, {}

    def evaluate(self, parameters, config):
        self.embedding.backward(torch.from_numpy(parameters))
        self.optimizer.step()
        self.optimizer.zero_grad()
        return 1, 1, {"accuracy"}


# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", client=FlowerClient(ClientModel(), load_data())
)
