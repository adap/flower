"""Collection of help functions needed by the strategies."""

import os
from threading import Lock
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from flwr.common import NDArrays, Parameters, Scalar, parameters_to_ndarrays
from natsort import natsorted
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .client import set_params, set_sklearn_model_params
from .dataset import get_cifar_10, get_partitioned_house, get_partitioned_income
from .models import CifarNet, MnistNet, test_cifar, test_mnist

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


def evaluate_aggregated(
    evaluate_fn: Optional[
        Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]
    ],
    server_round: int,
    parameters: Parameters
):
    """Evaluate model parameters using an evaluation function."""
    if evaluate_fn is None:
        # No evaluation function provided
        return None
    parameters_ndarrays = parameters_to_ndarrays(parameters)
    eval_res = evaluate_fn(server_round, parameters_ndarrays)
    if eval_res is None:
        return None
    loss, metrics = eval_res

    return loss, metrics


def mnist_evaluate(
    server_round: int, parameters: NDArrays
):
    """Evaluate MNIST model on the test set."""
    # determine device
    device = torch.device("cpu")

    model = MnistNet()
    set_params(model, parameters)
    model.to(device)

    testset = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)
    loss, accuracy, auc = test_mnist(model, testloader, device=device)

    return loss, {"accuracy": accuracy, "auc": auc}


def cifar_evaluate(
    server_round: int, parameters: NDArrays
):
    """Evaluate CIFAR-10 model on the test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CifarNet()
    set_params(model, parameters)
    model.to(device)

    _, testset = get_cifar_10()
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)
    loss, accuracy, auc = test_cifar(model, testloader, device=device)

    # return statistics
    return loss, {"accuracy": accuracy, "auc": auc}


def income_evaluate(
    server_round: int, parameters: NDArrays
):
    """Evaluate Income model on the test set."""
    model = LogisticRegression()
    model = set_sklearn_model_params(model, parameters)
    model.classes_ = np.array([0.0, 1.0])

    _, x_test, _, y_test = get_partitioned_income(
        "flanders/datasets_files/adult_server.csv", 1, train_size=0.0, test_size=1.0
    )
    x_test = x_test[0]
    y_test = y_test[0]
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, model.predict_proba(x_test))
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])

    return loss, {"accuracy": accuracy, "auc": auc}


def house_evaluate(
    server_round: int, parameters: NDArrays
):
    """Evaluate House model on the test set."""
    model = ElasticNet(alpha=1, warm_start=True)
    model = set_sklearn_model_params(model, parameters)
    _, x_test, _, y_test = get_partitioned_house(
        "flanders/datasets_files/houses_server.csv", 1, train_size=0.0, test_size=1.0
    )
    x_test = x_test[0]
    y_test = y_test[0]
    y_pred = model.predict(x_test)
    loss = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    rsq = r2_score(y_test, y_pred)
    arsq = 1 - (1 - rsq) * (len(y_test) - 1) / (
        len(y_test) - x_test.shape[1] - 1
    )  # noqa

    return loss, {"Adj-R2": arsq, "MAPE": mape}
