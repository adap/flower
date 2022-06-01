import argparse
import multiprocessing as mp
import sys
from collections import OrderedDict
from functools import partial

from loguru import logger

sys.path.insert(0, "../../src/py")

from typing import Callable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

import flwr as fl
from flwr.client.dp_client import DPClient, test
from flwr.server.strategy import FedAvgDp


# Training Model that is being used by the client
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.gn1 = nn.GroupNorm(int(6 / 3), 6)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.gn2 = nn.GroupNorm(int(16 / 4), 16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.gn1(x)
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.gn2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# function to load the MNIST dataset
XY = Tuple[np.ndarray, np.ndarray]
XYList = List[XY]
PartitionedDataset = List[Tuple[XY, XY]]


# def load(
#     num_partitions: int,
# ) -> PartitionedDataset:
#     """Create partitioned version of MNIST."""
#     xy_train, xy_test = tf.keras.datasets.mnist.load_data()
#     xy_train_partitions = create_partitions(xy_train, num_partitions)
#     xy_test_partitions = create_partitions(xy_test, num_partitions)
#     return list(zip(xy_train_partitions, xy_test_partitions))


def load_data(batch_size: int):
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True), DataLoader(
        testset
    )


# Function to get the weights of the model
def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(module: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set the PyTorch module parameters from a list of NumPy arrays."""
    params_dict = zip(module.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    module.load_state_dict(state_dict, strict=True)
    return module


# Main function to create DP client
def start_client(batch_size, epochs, cid) -> None:
    """Start a client."""
    module = Net()

    train_loader, test_loader = load_data(batch_size)

    optimizer = SGD(module.parameters(), lr=0.001)
    criterion = CrossEntropyLoss()
    target_epsilon = 1.0
    target_delta = 0.1
    privacy_engine = PrivacyEngine()

    client = DPClient(
        module,
        optimizer,
        criterion,
        privacy_engine,
        train_loader,
        test_loader,
        target_epsilon,
        target_delta,
        epochs=epochs,
        max_grad_norm=1.0,
    )
    logger.info("Starting client # {}", cid)
    fl.client.start_numpy_client("[::]:8080", client=client)


def set_parameters(self, parameters: List[np.ndarray]) -> None:
    """Set the PyTorch module parameters from a list of NumPy arrays."""
    params_dict = zip(self.module.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    self.module.load_state_dict(state_dict, strict=True)


def accuracy(predictions, actuals):
    total = actuals.size(0)
    correct = (torch.max(predictions, 1)[1].eq(actuals)).sum().item()
    return correct / total


def evaluate(
    module: nn.Module,
    criterion: Callable,
    dataloader: DataLoader,
    device: str,
    weights: List[np.ndarray],
):
    """Evaluation function for server side.

    Parameters
    ----------
    weights
        Updated model weights to evaluate.

    Returns
    -------
    loss
        Loss on the test set.
    accuracy
        Accuracy on the test set.
    """
    set_weights(module, weights)
    loss, _, metrics = test(module, criterion, dataloader, device, accuracy=accuracy)
    acc = metrics["accuracy"]
    logger.info("Accuracy: {}", acc)
    return float(loss), {"accuracy": float(acc)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument(
        "--num-clients",
        default=2,
        type=int,
        help="Total number of fl participants, requied to get correct partition",
    )
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        help="Data Partion to train on. Must be less than number of clients",
    )
    parser.add_argument(
        "--local-epochs",
        default=1,
        type=int,
        help="Total number of local epochs to train",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--learning-rate", default=0.15, type=float, help="Learning rate for training"
    )
    # DPSGD specific arguments
    parser.add_argument(
        "--dpsgd",
        default=False,
        type=bool,
        help="If True, train with DP-SGD. If False, " "train with vanilla SGD.",
    )
    parser.add_argument("--l2-norm-clip", default=1.0, type=float, help="Clipping norm")
    parser.add_argument(
        "--noise-multiplier",
        default=1.1,
        type=float,
        help="Ratio of the standard deviation to the clipping norm",
    )
    parser.add_argument(
        "--microbatches",
        default=32,
        type=int,
        help="Number of microbatches " "(must evenly divide batch_size)",
    )

    parser.add_argument(
        "-r", type=int, default=3, help="Number of rounds for the federated training"
    )

    parser.add_argument(
        "-nbc",
        type=int,
        default=2,
        help="Number of clients to keep track of dataset share",
    )

    parser.add_argument("-b", type=int, default=256, help="Batch size")

    parser.add_argument(
        "-fc",
        type=int,
        default=2,
        help="Min fit clients, min number of clients to be sampled next round",
    )

    parser.add_argument(
        "-ac",
        type=int,
        default=2,
        help="Min available clients, min number of clients that need to connect to the server before training round can start",
    )
    args = parser.parse_args()
    epochs = int(args.local_epochs)
    num_clients = int(args.num_clients)
    rounds = int(args.r)
    fc = int(args.fc)
    ac = int(args.ac)
    batch_size = int(args.b)

    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")
    # Create a new fresh model to initialize parameters
    net = Net()
    init_weights = get_weights(net)
    # Convert the weights (np.ndarray) to parameters
    init_param = fl.common.weights_to_parameters(init_weights)
    _, test_loader = load_data(batch_size)
    eval_fn = partial(evaluate, net, CrossEntropyLoss(), test_loader, "cpu")
    # Define the strategy
    strategy = FedAvgDp(
        fraction_fit=float(fc / ac),
        min_fit_clients=fc,
        min_available_clients=ac,
        eval_fn=eval_fn,
        initial_parameters=init_param,
    )
    server_process = mp.Process(
        target=fl.server.start_server,
        args=["[::]:8080"],
        kwargs=dict(config={"num_rounds": rounds}, strategy=strategy),
    )
    server_process.start()
    client_fn = partial(start_client, batch_size, epochs)
    with mp.Pool(num_clients) as pool:
        pool.map(client_fn, range(num_clients))
    server_process.kill()
