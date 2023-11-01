from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn

from task import load_data


class ClientModel(nn.Module):
    def __init__(self):
        super(ClientModel, self).__init__()
        self.fc = nn.Linear(5, 10)  # Simple Linear layer

    def forward(self, x):
        return self.fc(x)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, data):
        super(FlowerClient, self).__init__()
        self.data = data
        self.model = model
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        self.embedding = self.model(self.data[0][0])
        return [self.embedding.detach().numpy()], 1, {}

    def evaluate(self, parameters, config):
        self.embedding.backward(torch.from_numpy(parameters[0]))
        self.optimizer.step()
        self.optimizer.zero_grad()
        return 1.0, 1, {"accuracy": 0.0}


# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(ClientModel(), load_data()).to_client(),
)
