"""Contains utility functions for CNN FL on MNIST."""

import pickle
import torch
from torch import nn
from functools import reduce
from pathlib import Path
from secrets import token_hex
from typing import Dict, Optional, Union, List

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History


def apply_nn_compression(net: nn.Module, mask: torch.tensor) -> nn.Module:
    """Function to zero out some of the model weights.

    Parameters
    ----------
    net : nn.Module
        Model to be compressed.
    mask: torch.Tensor
        One dimensional binary vector having ones for weights that are preserved.
    """

    list_of_reshaped_layers = []
    list_of_shapes = []

    for layer in net.parameters():
        reshaped_layer = torch.flatten(layer.data)
        list_of_reshaped_layers.append(reshaped_layer)
        shape = reduce((lambda x, y: x * y), list(layer.data.shape))
        list_of_shapes.append(shape)
    cat_full_vec = torch.cat(list_of_reshaped_layers)
    compressed_full_vec = torch.mul(cat_full_vec, mask)

    compressed_split_vec = torch.split(compressed_full_vec, list_of_shapes)

    for i, layer in enumerate(net.parameters()):
        layer.data = compressed_split_vec[i].reshape(layer.data.shape)

    return net


def plot_metric_from_histories(
    hist: List[History],
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : Histories
        Lists of objects containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    accuracies_across_runs = []
    losses_across_runs = []
    rounds = None

    for i in range(len(hist)):
        rounds, accuracy_values = zip(*hist[i].metrics_centralized["accuracy"])
        _, loss_values = zip(*hist[i].losses_centralized)
        accuracies_across_runs.append(accuracy_values)
        losses_across_runs.append(loss_values)

    x_axis = range(rounds + 1)

    lowest_loss_across_runs = np.min(losses_across_runs, axis=0)
    largest_loss_across_runs = np.max(losses_across_runs, axis=0)
    lowest_accuracy_across_runs = np.min(accuracies_across_runs, axis=0)
    highest_accuracy_across_runs = np.max(accuracies_across_runs, axis=0)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="row", dpi=300)
    axs[0].fill_between(x_axis, largest_loss_across_runs, lowest_loss_across_runs, alpha=0.4)
    axs[1].fill_between(x_axis, highest_accuracy_across_runs, lowest_accuracy_across_runs, alpha=0.4)

    axs[0].set_ylabel("Loss")
    axs[1].set_ylabel("Accuracy")

    axs[1].set_ylim(bottom=0, top=1)
    axs[0].set_yscale('log')

    plt.xlabel("Rounds")

    plt.tight_layout()

    plt.savefig(Path(save_plot_path) / Path(f"centralized_metrics{suffix}.png"))
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
