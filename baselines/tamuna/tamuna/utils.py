"""Contains utility functions for CNN FL on MNIST."""

import pickle
import torch
from torch import nn
from functools import reduce
from pathlib import Path
from typing import List
from omegaconf import DictConfig

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


def save_results_as_pickle(
    history: History,
    file_path: str,
) -> None:
    """Saves results from simulation using pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: str
        Path to file to create and store history.
    """

    data = {"history": history}

    # save results to pickle
    with open(file_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def compare_histories(
    tamuna_histories: List[History],
    fedavg_histories: List[History],
    dim: int,
    save_path: str,
    cfg: DictConfig,
):
    compare_loss_and_accuracy(fedavg_histories, tamuna_histories, save_path)
    compare_communication_complexity(
        fedavg_histories, tamuna_histories, save_path, dim, cfg
    )


def compare_communication_complexity(
    fedavg_histories: List[History],
    tamuna_histories: List[History],
    save_path: str,
    dim: int,
    cfg: DictConfig,
):
    up_complexities = [
        np.ceil((cfg.server.s * dim) / cfg.server.clients_per_round),
        dim,
    ]
    down_complexities = [dim, dim]
    labels = ["Tamuna", "FedAvg"]

    for j, hist in enumerate([tamuna_histories, fedavg_histories]):
        accuracies_across_runs = []
        losses_across_runs = []
        rounds = None

        for i in range(len(hist)):
            rounds, accuracy_values = zip(*hist[i].metrics_centralized["accuracy"])
            _, loss_values = zip(*hist[i].losses_centralized)
            accuracies_across_runs.append(accuracy_values)
            losses_across_runs.append(loss_values)

        x_axis = np.arange(len(rounds)) * (
            cfg.server.uplink_factor * up_complexities[j]
            + cfg.server.downlink_factor * down_complexities[j]
        )

        lowest_loss_across_runs = np.min(losses_across_runs, axis=0)
        hishest_loss_across_runs = np.max(losses_across_runs, axis=0)

        avg_loss_across_runs = (
            np.add(lowest_loss_across_runs, hishest_loss_across_runs) / 2
        )

        plt.fill_between(
            x_axis,
            hishest_loss_across_runs,
            lowest_loss_across_runs,
            alpha=0.4,
            label=labels[j],
        )

        plt.plot(x_axis, avg_loss_across_runs, linewidth=2)

    plt.ylabel("Loss")
    plt.yscale("log")
    plt.xlabel("Communicated real numbers")
    plt.legend()
    plt.tight_layout()
    plt.savefig(Path(save_path) / Path(f"communication_complexity.png"))
    plt.close()


def compare_loss_and_accuracy(
    fedavg_histories: List[History], tamuna_histories: List[History], save_path: str
):
    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="row", dpi=300)
    labels = ["Tamuna", "FedAvg"]

    for j, hist in enumerate([tamuna_histories, fedavg_histories]):
        accuracies_across_runs = []
        losses_across_runs = []
        rounds = None

        for i in range(len(hist)):
            rounds, accuracy_values = zip(*hist[i].metrics_centralized["accuracy"])
            _, loss_values = zip(*hist[i].losses_centralized)
            accuracies_across_runs.append(accuracy_values)
            losses_across_runs.append(loss_values)

        x_axis = range(len(rounds))

        lowest_loss_across_runs = np.min(losses_across_runs, axis=0)
        highest_loss_across_runs = np.max(losses_across_runs, axis=0)
        lowest_accuracy_across_runs = np.min(accuracies_across_runs, axis=0)
        highest_accuracy_across_runs = np.max(accuracies_across_runs, axis=0)

        avg_loss_across_runs = (
            np.add(lowest_loss_across_runs, highest_loss_across_runs) / 2
        )
        avg_accuracy_across_runs = (
            np.add(lowest_accuracy_across_runs, highest_accuracy_across_runs) / 2
        )

        axs[0].fill_between(
            x_axis,
            highest_loss_across_runs,
            lowest_loss_across_runs,
            alpha=0.4,
            label=labels[j],
        )
        axs[0].plot(x_axis, avg_loss_across_runs, linewidth=2)

        axs[1].fill_between(
            x_axis,
            highest_accuracy_across_runs,
            lowest_accuracy_across_runs,
            alpha=0.4,
            label=labels[j],
        )
        axs[1].plot(x_axis, avg_accuracy_across_runs, linewidth=2)

    axs[0].set_ylabel("Loss")
    axs[0].set_yscale("log")
    axs[0].legend()

    axs[1].set_ylabel("Accuracy")
    axs[1].set_ylim(bottom=0, top=1)
    axs[1].legend()

    plt.xlabel("Rounds")
    plt.tight_layout()
    plt.savefig(Path(save_path) / Path(f"loss_accuracy.png"))
    plt.close()
