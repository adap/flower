import argparse
from collections import OrderedDict

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flamby.datasets.fed_heart_disease import NUM_CLIENTS, FedHeartDisease
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader as dl


# Define Flower client
class Client(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, cpu_only) -> None:
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.use_gpu = torch.cuda.is_available() and not (cpu_only)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, epochs=1, use_gpu=self.use_gpu)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, self.use_gpu)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def get_data(cid, train=True):
    return dl(
        FedHeartDisease(center=cid, train=train),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )


class Baseline(nn.Module):
    def __init__(self, input_dim=13, output_dim=1):
        super(Baseline, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


class BaselineLoss(_Loss):
    def __init__(self):
        super(BaselineLoss, self).__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.bce(input, target)


def train(net, trainloader, epochs, use_gpu):
    """Train the model on the training set."""
    loss = BaselineLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    for _ in range(epochs):
        for X, y in trainloader:
            if use_gpu:
                X = X.cuda()
                y = y.cuda()

            optimizer.zero_grad()
            y_pred = net(X)
            lm = loss(y_pred, y)
            lm.backward()
            optimizer.step()


def test(net, testloader, use_gpu):
    """Validate the model on the test set."""
    if use_gpu:
        net = net.cuda()
    net.eval()
    criterion = BaselineLoss()
    loss = 0.0
    with torch.no_grad():
        y_pred_final = []
        y_true_final = []
        for X, y in testloader:
            if use_gpu:
                X = X.cuda()
                y = y.cuda()
            y_pred = net(X).detach().cpu()
            y = y.detach().cpu()
            loss += criterion(y_pred, y).item()
            y_pred_final.append(y_pred.numpy())
            y_true_final.append(y.numpy())

        y_true_final = np.concatenate(y_true_final)
        y_pred_final = np.concatenate(y_pred_final)
    net.train()
    return loss / len(testloader.dataset), metric(y_true_final, y_pred_final)


def metric(y_true, y_pred):
    y_true = y_true.astype("uint8")
    # The try except is needed because when the metric is batched some batches
    # have one class only
    try:
        # return roc_auc_score(y_true, y_pred)
        # proposed modification in order to get a metric that calcs on center 2
        # (y=1 only on that center)
        return ((y_pred > 0.5) == y_true).mean()
    except ValueError:
        return np.nan


def gen_eval_fun(cpu_only):
    net = Baseline()
    data_loader = get_data(0, train=False)

    use_gpu = torch.cuda.is_available() and not (cpu_only)

    def eval_fun(round, params, config):
        params_dict = zip(net.state_dict().keys(), params)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

        loss, _ = test(net, data_loader, use_gpu)

        return loss, {}

    return eval_fun


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}


def get_client_fn(cpu_only):
    def client_fn(cid):
        net = Baseline()
        train_data = get_data(int(cid), train=True)
        test_data = get_data(int(cid), train=False)
        return Client(net, train_data, test_data, cpu_only)

    return client_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_rounds",
        type=int,
        default=3,
        help="Number of rounds to run FL for.",
    )
    parser.add_argument("--cpu_only", action="store_true")
    args = parser.parse_args()

    hist = fl.simulation.start_simulation(
        client_fn=get_client_fn(args.cpu_only),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=args.n_rounds),
        strategy=fl.server.strategy.FedAvg(
            evaluate_fn=gen_eval_fun(args.cpu_only),
            evaluate_metrics_aggregation_fn=weighted_average,
        ),
        ray_init_args={"logging_level": "error", "log_to_driver": False},
    )
