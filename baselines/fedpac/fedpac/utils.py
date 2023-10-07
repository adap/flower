"""Contains utility functions for CNN FL on MNIST."""

import pickle
import torch

from pathlib import Path
from secrets import token_hex
from functools import reduce
from itertools import pairwise
import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

from flwr.server.history import History
from flwr.common import  NDArrays
from omegaconf import  OmegaConf

from typing import Dict, Optional, Union, List, Tuple


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "distributed"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])

    rounds_loss, values_loss = zip(*hist.losses_distributed)
    print(rounds_loss, values_loss)
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    # axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss[0]['loss']))
    axs[1].plot(np.asarray(rounds), np.asarray(values))

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    # plt.title(f"{metric_type.capitalize()} Validation - MNIST")
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    extra_results: Optional[Dict] = {},
    default_filename: Optional[str] = "results.pkl",
) -> None:
    """Saves results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """

    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Adds a randomly generated suffix to the file name (so it doesn't
        overwrite the file)."""
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Appends the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        # file exists already
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history, **extra_results}

    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_on_fit_config(config):
    def fit_config_fn(server_round: int, global_centroid):
        # resolve and convert to python dict
        fit_config = OmegaConf.to_container(config, resolve=True)
        fit_config["curr_round"] = server_round             # add round info
        fit_config.update({"global_centroid":global_centroid, 
                            "classifier_head": None
                        }) 
        return fit_config

    return fit_config_fn

def get_on_fit_fedavg_config(config):
    def fit_config_fn(server_round: int):
        # resolve and convert to python dict
        fit_config = OmegaConf.to_container(config, resolve=True)
        fit_config["curr_round"] = server_round             # add round info
        return fit_config

    return fit_config_fn

def get_centroid(feature_list: List[torch.FloatTensor]):
    """Takes in the feature extraction layers list 
    and returns mean of feature extraction layers of each labels"""
    features_mean = {}
    for [label, feat_list] in feature_list.items():
        feat = 0 * feat_list[0]
        for i in feat_list:
            feat += i
        features_mean[label] = feat / len(feat_list)

    return features_mean

def aggregate_weights(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate_centroids(centroids_list, class_sizes):
    "Compute estimated global centroid"
    aggregated_centroids_list, aggregated_class_sizes = {}, {}
    for i in range(len(centroids_list)):
        for label, centroid in centroids_list[i].items():
            if label in aggregated_centroids_list.keys():
                aggregated_centroids_list[label].append(centroid)
                aggregated_class_sizes[label].append(class_sizes[i][label])
            else:
                aggregated_centroids_list[label] = [centroid]
                aggregated_class_sizes[label] = [class_sizes[i][label]]
    
    for label, centroid in aggregated_centroids_list.items():
        size_list = aggregated_class_sizes[label]
        c = torch.zeros(len(centroid[0]))
        for i in range(len(centroid)):
            c += centroid[i]*size_list[i]
        aggregated_centroids_list[label] = c/sum(size_list).item()

    return aggregated_centroids_list
    

def aggregate_heads(stats, device):
    var = [s[0] for s in stats]
    bias = [s[1] for s in stats]

    num_clients = len(var)
    num_class = bias[0].shape[0]
    d = bias[0].shape[1]
    agg_head = []
    bias = torch.stack(bias).to(device)


    for i in range(num_clients):
        v = torch.tensor(var, device=device)
        href = bias[i]
        dist = torch.zeros((num_clients, num_clients), device=device)

        for j, k in pairwise(tuple(range(num_clients))):
            hj = bias[j]
            hk = bias[k]
            h = torch.zeros((d, d), device=device)
            for m in range(num_class):
                h+=torch.mm((href[m]-hj[m]).reshape(d, 1), (href[m]-hk[m]).reshape(1, d))
            
            d_jk = torch.trace(h)
            dist[j][k] = d_jk
            dist[k][j] = d_jk

        p_matrix = torch.diag(v) + dist
        evals, evecs = torch.linalg.eig(p_matrix)

        p_matrix = p_matrix.cpu().numpy()  # coefficient for QP problem

        p_matrix_new = 0
        for x in range(num_clients):
            if evals.real[x] >= 0.01:
                p_matrix_new += evals[x]*torch.mm(evecs[:,x].reshape(num_clients, 1), evecs[:,x].reshape(1, num_clients))
                p_matrix = p_matrix_new.cpu().numpy() if not np.all(np.linalg.eigvals(p_matrix)>=0.0) else p_matrix

        alpha = 0
        eps = 1e-3

        if np.all(np.linalg.eigvals(p_matrix)>=0):
            alphav = cvx.Variable(num_clients)
            obj = cvx.Minimize(cvx.quad_form(alphav, p_matrix))
            prob = cvx.Problem(obj, [cvx.sum(alphav) == 1.0, alphav >= 0])
            prob.solve()
            alpha = alphav.value
            alpha = [(a)*(a>eps) for a in alpha] # zero-out small weights (<eps)

        else:
            alpha = None
        
        agg_head.append(alpha)

    return agg_head




