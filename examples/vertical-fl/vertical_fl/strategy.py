import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays


class ServerModel(nn.Module):
    def __init__(self, input_size):
        super(ServerModel, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        return self.sigmoid(x)


class Strategy(fl.server.strategy.FedAvg):
    def __init__(self, labels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = ServerModel(12)
        self.initial_parameters = ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        )
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.BCELoss()
        self.label = torch.tensor(labels).float().unsqueeze(1)

    def aggregate_fit(
        self,
        rnd,
        results,
        failures,
    ):
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        embedding_results = [
            torch.from_numpy(parameters_to_ndarrays(fit_res.parameters)[0])
            for _, fit_res in results
        ]
        embeddings_aggregated = torch.cat(embedding_results, dim=1)
        embedding_server = embeddings_aggregated.detach().requires_grad_()
        output = self.model(embedding_server)
        loss = self.criterion(output, self.label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        grads = embedding_server.grad.split([4, 4, 4], dim=1)
        np_grads = [grad.numpy() for grad in grads]
        parameters_aggregated = ndarrays_to_parameters(np_grads)

        with torch.no_grad():
            correct = 0
            output = self.model(embedding_server)
            predicted = (output > 0.5).float()

            correct += (predicted == self.label).sum().item()

            accuracy = correct / len(self.label) * 100

        metrics_aggregated = {"accuracy": accuracy}

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        return None, {}
