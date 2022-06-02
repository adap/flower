import argparse
import multiprocessing as mp
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from collections import OrderedDict
from functools import partial

from loguru import logger

sys.path.insert(0, "../../src/py")

from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor

import flwr as fl
from flwr.client.dp_client import DPClient, test
from flwr.common.typing import Parameters
from flwr.server.strategy import FedAvgDp


class Net(nn.Module):
    """An example convolutional neural network for fitting the CIFAR-10 dataset."""

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
        """Neural network forward pass."""
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.gn1(x)
        x = self.pool(torch.tanh(self.conv2(x)))
        x = self.gn2(x)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


def load_data(batch_size: int, num_clients: int = 1, cid: int = 0):
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    train_indices = np.array_split(np.arange(len(trainset.data)), num_clients)[cid]
    test_indices = np.array_split(np.arange(len(testset.data)), num_clients)[cid]
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(testset, test_indices)
    return DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, drop_last=True
    ), DataLoader(test_subset, batch_size=batch_size)


def get_weights(model):
    """Convert PyTorch module parameters to a numpy array."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(module: nn.Module, parameters: List[np.ndarray]) -> None:
    """Set the PyTorch module parameters from a list of NumPy arrays."""
    params_dict = zip(module.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    module.load_state_dict(state_dict, strict=True)
    return module


def accuracy(predictions, actuals):
    """Multi-class classification accuracy function."""
    total = actuals.size(0)
    correct = (torch.max(predictions, 1)[1].eq(actuals)).sum().item()
    return correct / total


# Main function to create DP client
def start_client(
    batch_size: int,
    epochs: int,
    rounds: int,
    num_clients: int,
    device: str,
    target_epsilon: float,
    max_grad_norm: float,
    learning_rate: float,
    cid: int,
) -> None:
    """Start a client."""
    module = Net()

    train_loader, test_loader = load_data(batch_size, num_clients, cid)

    optimizer = SGD(module.parameters(), lr=learning_rate)
    criterion = CrossEntropyLoss()
    target_epsilon = 1.0
    target_delta = 1 / len(train_loader.dataset)
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
        rounds=rounds,
        max_grad_norm=max_grad_norm,
        device=device,
        cid=cid,
        accuracy=accuracy,
    )
    logger.info("Starting client # {}", cid)
    fl.client.start_numpy_client("[::]:8080", client=client)


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
    logger.info("Global model accuracy: {}", acc)
    return float(loss), {"accuracy": float(acc)}


def start_server(init_param: Parameters, fc: int, ac: int, rounds: int, eval_fn: Callable):
    """Start the Flower server with the differential privacy strategy."""
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
    return server_process


def make_arg_parser() -> argparse.ArgumentParser:
    """Get a command line argument parser."""
    parser = argparse.ArgumentParser(description="Flower Differential Privacy Demo")
    parser.add_argument(
        "--num-clients",
        default=2,
        type=int,
        help="Total number of clients",
    )
    parser.add_argument(
        "--epochs",
        default=1,
        type=int,
        help="Total number of local epochs to train",
    )
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument(
        "--learning-rate", default=0.01, type=float, help="Learning rate for training"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1.0,
        help="Target epsilon for the privacy budget.",
    )
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Gradient clipping norm")

    parser.add_argument(
        "--rounds", type=int, default=3, help="Number of rounds for the federated training"
    )

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
    return parser


if __name__ == "__main__":
    args = make_arg_parser().parse_args()
    epochs = int(args.epochs)
    num_clients = int(args.num_clients)
    rounds = int(args.rounds)
    fc = int(args.fc)
    ac = int(args.ac)
    target_epsilon = float(args.eps)
    batch_size = int(args.batch_size)
    max_grad_norm = float(args.max_grad_norm)
    learning_rate = float(args.learning_rate)
    # Set the start method for multiprocessing in case Python version is under 3.8.1
    mp.set_start_method("spawn")
    # Create a new fresh model to initialize parameters
    net = Net()
    init_weights = get_weights(net)
    # Convert the weights (np.ndarray) to parameters
    init_param = fl.common.weights_to_parameters(init_weights)
    _, test_loader = load_data(batch_size)
    eval_fn = partial(evaluate, net, CrossEntropyLoss(), test_loader, "cpu")
    server_process = start_server(init_param, fc, ac, rounds, eval_fn)
    client_fn = partial(
        start_client, batch_size, epochs, rounds, num_clients, "cpu", target_epsilon, max_grad_norm, learning_rate
    )
    with mp.Pool(num_clients) as pool:
        pool.map(client_fn, range(num_clients))
    server_process.kill()
