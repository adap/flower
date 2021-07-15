import flwr as fl
import ray
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple
from dataset_utils import getCIFAR10, do_fl_partitioning, get_dataloader


# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
# borrowed from Pytorch quickstart example
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# borrowed from Pytorch quickstart example
def train(net, trainloader, epochs, device: str):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


# borrowed from Pytorch quickstart example
def test(net, testloader, device: str):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


# Flower client that will be spawned by Ray
# Adapted from Pytorch quickstart example
class CifarRayClient(fl.client.NumPyClient):

    def __init__(self, cid: str, fed_dir_data: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)

        # instantiate model
        self.net = Net()

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):

        print(f"fit() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        trainloader = get_dataloader(self.fed_dir, self.cid, is_train=True,
                                     batch_size=int(config["batch_size"]),
                                     workers=num_workers)

        # send model to device
        self.net.to(self.device)

        # train
        train(self.net, trainloader, epochs=int(config["epochs"]), device=self.device)

        # return local model and statistics
        return self.get_parameters(), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):

        print(f"evaluate() on client cid={self.cid}")
        self.set_parameters(parameters)

        # load data for this client and get trainloader
        num_workers = len(ray.worker.get_resource_ids()["CPU"])
        valloader = get_dataloader(self.fed_dir, self.cid, is_train=False,
                                   batch_size=50,
                                   workers=num_workers)

        # send model to device
        self.net.to(self.device)

        # evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(rnd: int) -> Dict[str, str]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epoch_global": str(rnd),
        "epochs": str(5),
        "batch_size": str(64),
    }
    return config


def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)


def get_eval_fn(
    testset: torchvision.datasets.CIFAR10,
) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(weights: fl.common.Weights) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = Net()
        set_weights(model, weights)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Ray simulation (a _default server_ will be created)
if __name__ == "__main__":

    pool_size = 100  # number of partions (= number of total clients in the experiment)
    client_resources = {'num_cpus': 1}  # each client will get allocated 1 CPUs

    # download CIFAR10 dataset
    train_path, testset = getCIFAR10()

    # partition dataset
    fed_dir = do_fl_partitioning(train_path, pool_size=pool_size, alpha=1000,
                                 num_classes=10, val_ratio=0.1)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        min_fit_clients=10,
        min_available_clients=pool_size,
        on_fit_config_fn=fit_config,
        eval_fn=get_eval_fn(testset),
    )

    # (optional) specify ray config
    ray_config = {'include_dashboard': False}
    fl.simulation.start_ray_simulation(pool_size, fed_dir, client_resources,
                                       CifarRayClient, config={"num_rounds": 20},
                                       ray_init_config=ray_config,
                                       strategy=strategy)
