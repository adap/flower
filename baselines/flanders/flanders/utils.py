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


def save_params(parameters, cid, dir="clients_params", remove_last=False, rrl=False):
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
    path_file = f"{dir}/{cid}_params.npy"
    if os.path.exists(dir) is False:
        os.mkdir(dir)
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


def save_results(loss, accuracy, config=None, output_dir="results"):
    """Save results in a csv file.

    Parameters
    ----------
    loss : float
        Loss value.
    accuracy : float
        Accuracy value.
    config : dict, optional
        Configurations to save, by default None.
    output_dir : str, optional
        Output directory, by default "results".
    """
    # Generate csv
    config["accuracy"] = accuracy
    config["loss"] = loss
    df = pd.DataFrame.from_records([config])
    csv_path = f"{output_dir}/all_results.csv"
    # check that dir exists
    if os.path.exists(output_dir) is False:
        os.mkdir(output_dir)
    # Lock is needed when multiple clients are running concurrently
    with lock:
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False, header=True)


def load_all_time_series(dir="clients_params", window=0):
    """Load all time series.

    Load all time series in order to have a tensor of shape (m,T,n)
    where:
    - T := time;
    - m := number of clients;
    - n := number of parameters.
    """
    files = os.listdir(dir)
    files = natsorted(files)
    data = []
    for file in files:
        data.append(np.load(os.path.join(dir, file), allow_pickle=True))

    return np.array(data)[:, -window:, :]


def flatten_params(params):
    """Transform a list of (layers-)parameters into a single vector of shape (n)."""
    return np.concatenate(params, axis=None).ravel()


def evaluate_aggregated(
    evaluate_fn: Optional[
        Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]]]
    ],
    server_round: int,
    parameters: Parameters,
    config: Dict[str, Scalar],
):
    """Evaluate model parameters using an evaluation function."""
    if evaluate_fn is None:
        # No evaluation function provided
        return None
    parameters_ndarrays = parameters_to_ndarrays(parameters)
    eval_res = evaluate_fn(server_round, parameters_ndarrays, config)
    if eval_res is None:
        return None
    loss, metrics = eval_res
    return loss, metrics


def mnist_evaluate(
    server_round: int, parameters: NDArrays, config: Dict[str, Scalar], output_dir: str
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

    config["round"] = server_round
    config["auc"] = auc
    save_results(loss, accuracy, config=config, output_dir=output_dir)

    return loss, {"accuracy": accuracy, "auc": auc}


def cifar_evaluate(
    server_round: int, parameters: NDArrays, config: Dict[str, Scalar], output_dir: str
):
    """Evaluate CIFAR-10 model on the test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CifarNet()
    set_params(model, parameters)
    model.to(device)

    _, testset = get_cifar_10()
    testloader = torch.utils.data.DataLoader(testset, batch_size=32)
    loss, accuracy, auc = test_cifar(model, testloader, device=device)

    config["round"] = server_round
    config["auc"] = auc
    save_results(loss, accuracy, config=config, output_dir=output_dir)

    # return statistics
    return loss, {"accuracy": accuracy, "auc": auc}


def income_evaluate(
    server_round: int, parameters: NDArrays, config: Dict[str, Scalar], output_dir: str
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

    config["round"] = server_round
    config["auc"] = auc
    save_results(loss, accuracy, config=config, output_dir=output_dir)

    return loss, {"accuracy": accuracy, "auc": auc}


def house_evaluate(
    server_round: int, parameters: NDArrays, config: Dict[str, Scalar], output_dir: str
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
    config["round"] = server_round
    config["auc"] = mape
    save_results(loss, arsq, config=config, output_dir=output_dir)

    return loss, {"Adj-R2": arsq, "MAPE": mape}
