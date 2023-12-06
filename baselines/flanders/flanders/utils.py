"""
Collection of help functions needed by the strategies.
"""

import numpy as np
import os
import json
import torch
import pandas as pd
from natsort import natsorted
from typing import Dict, Optional, Tuple, List
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    NDArrays,
)
from threading import Lock
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from .models import (
    MnistNet, 
    ToyNN, 
    roc_auc_multiclass, 
    test_toy, 
    train_mnist, 
    test_mnist, 
    train_toy
)
from .client import set_params

lock = Lock()            # if the script is run on multiple processors we need a lock to save the results

def l2_norm(true_matrix, predicted_matrix):
    """
    Compute the l2 norm between two matrices.

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
    anomaly_scores = np.sum(delta**2,axis=-1)**(1./2)
    return anomaly_scores

def save_params(parameters, cid, dir="clients_params", remove_last=False, rrl=False):
    """
    Args:
    - parameters (ndarray): decoded parameters to append at the end of the file
    - cid (int): identifier of the client
    - remove_last (bool): if True, remove the last saved parameters and replace with "parameters"
    - rrl (bool): if True, remove the last saved parameters and replace with the ones saved before this round
    """
    new_params = parameters
    # Save parameters in clients_params/cid_params
    path_file = f"{dir}/{cid}_params.npy"
    if os.path.exists(dir) == False:
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
    print(f"new_params shape of {cid}: {new_params.shape}")
    # save parameters
    np.save(path_file, new_params)


def save_predicted_params(parameters, cid):
    """
    Args:
    - parameters (ndarray): decoded parameters to append at the end of the file
    - cid (int): identifier of the client
    """
    new_params = parameters
    # Save parameters in client_params/cid_params
    path = f"strategy/clients_predicted_params/{cid}_params.npy"
    if os.path.exists("strategy/clients_predicted_params") == False:
        os.mkdir("strategy/clients_predicted_params")
    if os.path.exists(path):
        # load old parameters
        old_params = np.load(path, allow_pickle=True)
        # add new parameters
        new_params = np.vstack((old_params, new_params))
    # save parameters
    np.save(path, new_params)


def save_results(loss, accuracy, config=None):
    # Generate csv
    config["accuracy"] = accuracy
    config["loss"] = loss
    df = pd.DataFrame.from_records([config])
    csv_path = "results/all_results.csv"
    # check that dir exists
    if os.path.exists("results") == False:
        os.mkdir("results")
    # Lock is needed when multiple clients are running concurrently
    with lock:
        if os.path.exists(csv_path):
            df.to_csv(csv_path, mode="a", header=False, index=False)
        else:
            df.to_csv(csv_path, index=False, header=True)


def load_all_time_series(dir="clients_params", window=0):
        """
        Load all time series in order to have a tensor of shape (m,T,n)
        where:
        - T := time;
        - m := number of clients;
        - n := number of parameters
        """
        files = os.listdir(dir)
        files = natsorted(files)
        data = []
        for file in files:
            data.append(np.load(os.path.join(dir, file), allow_pickle=True))

        return np.array(data)[:,-window:,:]


def load_time_series(dir="", cid=0):
    """
    Load time series of client cid in order to have a matrix of shape (T,n)
    where:
    - T := time;
    - n := number of parameters
    """
    files = os.listdir(dir)
    files.sort()
    data = []
    for file in files:
        if file == f"{cid}_params.npy":
            data = np.load(os.path.join(dir, file), allow_pickle=True)
    return np.array(data)


def flatten_params(params):
    """
    Transform a list of (layers-)parameters into a single vector of shape (n).
    """
    return np.concatenate(params, axis=None).ravel()


def evaluate_aggregated(
    evaluate_fn, server_round: int, parameters: Parameters, config: Dict[str, Scalar]
) -> Optional[Tuple[float, Dict[str, Scalar]]]:
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
    server_round: int, parameters: NDArrays, config: Dict[str, Scalar]
):
    # determine device
    device = torch.device("cpu")

    model = MnistNet()
    set_params(model, parameters)
    model.to(device)

    testset = MNIST("", train=False, download=True, transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=1)
    loss, accuracy, auc = test_mnist(model, testloader, device=device)

    #config["id"] = args.exp_num
    config["round"] = server_round
    config["auc"] = auc
    save_results(loss, accuracy, config=config)
    print(f"Round {server_round} accuracy: {accuracy} loss: {loss} auc: {auc}")

    return loss, {"accuracy": accuracy, "auc": auc}
