import datetime
import os
import time
import warnings
from collections import OrderedDict

import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import flwr as fl
from flwr.common.serde import client_message_from_proto, server_message_from_proto

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


def get_tensorboard_middleware(logdir):
    os.makedirs(logdir, exist_ok=True)

    # To allow multiple runs and group those we will create a subdir
    # in the logdir which is named as number of directories in logdir + 1
    run_id = str(
        len(
            [
                name
                for name in os.listdir(logdir)
                if os.path.isdir(os.path.join(logdir, name))
            ]
        )
    )
    run_id = run_id + "-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    logdir_run = os.path.join(logdir, run_id)

    def tensorboard_middleware(fwd, app):
        start_time = None

        project_name = logdir
        workload_id = str(fwd.task_ins.workload_id)
        client_id = str(fwd.task_ins.task.consumer.node_id)
        group_id = str(fwd.task_ins.group_id)
        group_name = f"Workload ID: {workload_id}"
        run_name = f"Client ID: {client_id}"

        server_message = server_message_from_proto(
            fwd.task_ins.task.legacy_server_message
        )

        if server_message.fit_ins:
            config = server_message.fit_ins.config
            if "round" in config:
                round = config["round"]
            if "project" in config:
                project_name = str(config["project"])
            if "group" in config:
                group_name = str(config["group"])

            start_time = time.time()

        if server_message.evaluate_ins:
            config = server_message.evaluate_ins.config
            if "round" in config:
                round = config["round"]
            if "project" in config:
                project_name = str(config["project"])
            if "group" in config:
                group_name = str(config["group"])

        bwd = app(fwd)

        client_message = client_message_from_proto(
            bwd.task_res.task.legacy_client_message
        )

        writer = tf.summary.create_file_writer(os.path.join(logdir_run, client_id))

        # Write aggregated loss
        with writer.as_default(step=group_id):  # pylint: disable=not-context-manager
            if client_message.evaluate_res:
                tf.summary.scalar(
                    "eval_loss", client_message.evaluate_res.loss, step=group_id
                )
                if "accuracy" in client_message.evaluate_res.metrics:
                    tf.summary.scalar(
                        "eval_accuracy",
                        client_message.evaluate_res.metrics["accuracy"],
                        step=group_id,
                    )
            if client_message.fit_res:
                if start_time is not None:
                    tf.summary.scalar(
                        "fit_time", time.time() - start_time, step=group_id
                    )
                if "accuracy" in client_message.fit_res.metrics:
                    tf.summary.scalar(
                        "training_accuracy",
                        client_message.fit_res.metrics["accuracy"],
                        step=group_id,
                    )
                if "loss" in client_message.fit_res.metrics:
                    tf.summary.scalar(
                        "training_loss",
                        client_message.fit_res.metrics["loss"],
                        step=group_id,
                    )
            writer.flush()

        return bwd

    return tensorboard_middleware


# To run this: `flower-client --callable client:flower`
flower = fl.flower.Flower(
    client_fn=client_fn, middleware=[get_tensorboard_middleware(".runs_history")]
)


if __name__ == "__main__":
    # Start Flower client
    fl.client.start_client(
        server_address="0.0.0.0:9092",
        client=FlowerClient().to_client(),
        transport="grpc-rere",
    )
