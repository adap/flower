"""Collection of help functions needed by the strategies."""

import os
from threading import Lock
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import NDArrays, Parameters, Scalar, parameters_to_ndarrays
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST

from .client import set_params
from .models import FMnistNet, MnistNet, test_fmnist, test_mnist

lock = Lock()


def l2_norm(true_matrix, predicted_matrix):
    """Compute the l2 norm between two matrices.

    Parameters
    ----------
    true_matrix : ndarray
        The true matrix.
    predicted_matrix : ndarray
        The predicted matrix by MAR.

    Returns
    -------
    anomaly_scores : ndarray
        1-d array of anomaly scores.
    """
    delta = np.subtract(true_matrix, predicted_matrix)
    anomaly_scores = np.sum(delta**2, axis=-1) ** (1.0 / 2)
    return anomaly_scores


def save_params(
    parameters, cid, params_dir="clients_params", remove_last=False, rrl=False
):
    """Save parameters in a file.

    Args:
    - parameters (ndarray): decoded parameters to append at the end of the file
    - cid (int): identifier of the client
    - remove_last (bool):
        if True, remove the last saved parameters and replace with "parameters"
    - rrl (bool):
        if True, remove the last saved parameters and replace with the ones
        saved before this round.
    """
    new_params = parameters
    # Save parameters in clients_params/cid_params
    path_file = f"{params_dir}/{cid}_params.npy"
    if os.path.exists(params_dir) is False:
        os.mkdir(params_dir)
    if os.path.exists(path_file):
        # load old parameters
        old_params = np.load(path_file, allow_pickle=True)
        if remove_last:
            old_params = old_params[:-1]
            if rrl:
                new_params = old_params[-1]
        # add new parameters
        new_params = np.vstack((old_params, new_params))

    # save parameters
    np.save(path_file, new_params)


def load_all_time_series(params_dir="clients_params", window=0):
    """Load all time series.

    Load all time series in order to have a tensor of shape (m,T,n)
    where:
    - T := time;
    - m := number of clients;
    - n := number of parameters.
    """
    files = os.listdir(params_dir)
    files = natsorted(files)
    data = []
    for file in files:
        data.append(np.load(os.path.join(params_dir, file), allow_pickle=True))

    return np.array(data)[:, -window:, :]


def flatten_params(params):
    """Transform a list of (layers-)parameters into a single vector of shape (n)."""
    return np.concatenate(params, axis=None).ravel()


# pylint: disable=unused-argument
def evaluate_aggregated(
    evaluate_fn: Optional[
        Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]
    ],
    server_round: int,
    parameters: Parameters,
):
    """Evaluate model parameters using an evaluation function."""
    if evaluate_fn is None:
        # No evaluation function provided
        return None
    parameters_ndarrays = parameters_to_ndarrays(parameters)
    eval_res = evaluate_fn(server_round, parameters_ndarrays, {})
    if eval_res is None:
        return None
    loss, metrics = eval_res

    return loss, metrics


# pylint: disable=unused-argument
def mnist_evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
    """Evaluate MNIST model on the test set."""
    # determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = MnistNet()
    set_params(model, parameters)
    model.to(device)

    testset = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)
    loss, accuracy, auc = test_mnist(model, testloader, device=device)

    return loss, {"accuracy": accuracy, "auc": auc}


# pylint: disable=unused-argument
def fmnist_evaluate(server_round: int, parameters: NDArrays, config: Dict[str, Scalar]):
    """Evaluate MNIST model on the test set."""
    # determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = FMnistNet()
    set_params(model, parameters)
    model.to(device)

    testset = FashionMNIST(
        "", train=False, download=True, transform=transforms.ToTensor()
    )
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)
    loss, accuracy, auc = test_fmnist(model, testloader, device=device)

    return loss, {"accuracy": accuracy, "auc": auc}


def update_confusion_matrix(
    confusion_matrix: Dict[str, int],
    clients_states: Dict[str, bool],
    malicious_clients_idx: List,
    good_clients_idx: List,
):
    """Update TN, FP, FN, TP of confusion matrix."""
    for client_idx, client_state in clients_states.items():
        if int(client_idx) in malicious_clients_idx:
            if client_state:
                confusion_matrix["TP"] += 1
            else:
                confusion_matrix["FP"] += 1
        elif int(client_idx) in good_clients_idx:
            if client_state:
                confusion_matrix["FN"] += 1
            else:
                confusion_matrix["TN"] += 1
    return confusion_matrix
