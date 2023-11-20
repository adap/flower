"""
Collection of help functions needed by the strategies.
"""

import numpy as np
import os
import json
import pandas as pd
from natsort import natsorted
from typing import Dict, Optional, Tuple, List
from flwr.common import (
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)
from threading import Lock

lock = Lock()            # if the script is run on multiple processors we need a lock to save the results

def save_params(parameters, cid, remove_last=False, rrl=False):
    """
    Args:
    - parameters (ndarray): decoded parameters to append at the end of the file
    - cid (int): identifier of the client
    - remove_last (bool): if True, remove the last saved parameters and replace with "parameters"
    - rrl (bool): if True, remove the last saved parameters and replace with the ones saved before this round
    """
    new_params = parameters
    # Save parameters in client_params/cid_params
    path = f"strategy/clients_params/{cid}_params.npy"
    if os.path.exists("strategy/clients_params") == False:
        os.mkdir("strategy/clients_params")
    if os.path.exists(path):
        # load old parameters
        old_params = np.load(path, allow_pickle=True)
        if remove_last:
            old_params = old_params[:-1]
            if rrl:
                new_params = old_params[-1]
        # add new parameters
        new_params = np.vstack((old_params, new_params))
    # save parameters
    np.save(path, new_params)


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


def load_all_time_series(dir="", window=0):
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