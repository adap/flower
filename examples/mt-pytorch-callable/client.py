import datetime
import time
import warnings
from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
from flwr.common.serde import client_message_from_proto, server_message_from_proto
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

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
        return self.fc3(x)


def train(net, trainloader, epochs):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Load model and data (simple CNN, CIFAR-10)
net = Net().to(DEVICE)
trainloader, testloader = load_data()


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}


def client_fn(cid: str):
    """."""
    return FlowerClient().to_client()


def get_middleware(name):
    now = datetime.datetime.now().strftime("%b%d_%H_%M")
    wandb_group = f"exp_{now}"

    def wandb_middleware(fwd, app):
        start_time = None
        project_name = name
        group_name = wandb_group
        round = ""
        client_id = fwd.task_ins.task.consumer.node_id

        server_message = server_message_from_proto(
            fwd.task_ins.task.legacy_server_message
        )

        if server_message.fit_ins:
            config = server_message.fit_ins.config
            if "round" in config:
                round = f"_rnd-{config['round']}"
            if "project" in config:
                project_name = str(config["project"])
            if "group" in config:
                group_name = str(config["group"])
            start_time = time.time()
        if server_message.evaluate_ins:
            config = server_message.evaluate_ins.config
            if "round" in config:
                round = f"_rnd-{config['round']}"
            if "project" in config:
                project_name = str(config["project"])
            if "group" in config:
                group_name = str(config["group"])

        wandb.init(project=project_name, group=group_name, name=f"client-{client_id}")

        bwd = app(fwd)

        results_to_log = {}

        if len(round) > 0:
            results_to_log["round"] = round

        client_message = client_message_from_proto(
            bwd.task_res.task.legacy_client_message
        )

        if client_message.evaluate_res:
            results_to_log["evaluate_loss"] = client_message.evaluate_res.loss
            if "accuracy" in client_message.evaluate_res.metrics:
                results_to_log["accuracy"] = client_message.evaluate_res.metrics[
                    "accuracy"
                ]

        if client_message.fit_res:
            if start_time is not None:
                results_to_log["fit_time"] = time.time() - start_time
            if "loss" in client_message.fit_res.metrics:
                results_to_log["loss"] = client_message.fit_res.metrics["loss"]
            if "accuracy" in client_message.fit_res.metrics:
                results_to_log["accuracy"] = client_message.fit_res.metrics["accuracy"]

        wandb.log(results_to_log)

        return bwd

    return wandb_middleware


# To run this: `flower-client --callable client:flower`
flower = fl.flower.Flower(
    client_fn=client_fn, middleware=[get_middleware("MT PyTorch Callable")]
)


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:9092",
        client=FlowerClient().to_client(),
        transport="grpc-rere",
    )
