"""Contains utility functions for CNN FL on MNIST."""

import pickle
from functools import reduce
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr.server.history import History
from omegaconf import DictConfig
from torch import nn


def apply_nn_compression(net: nn.Module, mask: torch.tensor) -> nn.Module:
    """Zero out some of the model weights based on compression mask.

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
    """Save results from simulation using pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: str
        Path to file to create and store history.
    """
    data = {"history": history}

    # save results to pickle
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def compare_histories(
    tamuna_histories: List[History],
    fedavg_histories: List[History],
    dim: int,
    save_path: str,
    cfg: DictConfig,
):
    """Compare Tamuna and FedAvg histories."""
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
    """Compare Tamuna with FedAvg based on communication complexity."""
    plot_histories(
        tamuna_histories,
        cfg,
        dim,
        np.ceil((cfg.server.s * dim) / cfg.server.clients_per_round),
        "Tamuna",
    )
    plot_histories(fedavg_histories, cfg, dim, dim, "FedAvg")

    plt.ylabel("Loss")
    plt.yscale("log")
    plt.xlabel("Communicated real numbers")
    plt.legend()
    plt.grid(visible=True, which="both", linewidth=0.5, alpha=0.5)
    plt.minorticks_on()
    plt.title("Communication complexity")
    plt.tight_layout()
    plt.savefig(Path(save_path) / Path("communication_complexity.png"), dpi=300)
    plt.close()


def plot_histories(
    histories: List[History],
    cfg: DictConfig,
    down_complexity: int,
    up_complexity: int,
    label: str,
):
    """Plot multiple runs."""
    losses_across_runs = []

    for run in histories:
        history_losses = run.losses_centralized
        rounds, loss_values = zip(*history_losses)
        losses_across_runs.append(loss_values)

    x_axis = np.arange(len(rounds)) * (
        cfg.server.uplink_factor * up_complexity
        + cfg.server.downlink_factor * down_complexity
    )
    lowest_loss_across_runs = np.min(losses_across_runs, axis=0)
    highest_loss_across_runs = np.max(losses_across_runs, axis=0)
    avg_loss_across_runs = np.add(lowest_loss_across_runs, highest_loss_across_runs) / 2
    plt.fill_between(
        x_axis,
        highest_loss_across_runs,
        lowest_loss_across_runs,
        alpha=0.4,
        label=label,
    )
    plt.plot(x_axis, avg_loss_across_runs, linewidth=2)


def compare_loss_and_accuracy(
    fedavg_histories: List[History], tamuna_histories: List[History], save_path: str
):
    """Compare Tamuna with FedAvg based on loss and accuracy."""
    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")

    plot_histories_with_accuracy(axs, tamuna_histories, label="Tamuna")
    plot_histories_with_accuracy(axs, fedavg_histories, label="FedAvg")

    axs[0].set_title("MNIST Test Loss")
    axs[0].set_ylabel("Loss")
    axs[0].set_yscale("log")
    axs[0].legend()
    axs[0].grid(visible=True, which="both", linewidth=0.5, alpha=0.5)
    axs[0].minorticks_on()
    axs[0].set_xlabel("Rounds")

    axs[1].set_title("MNIST Test Accuracy")
    axs[1].set_ylabel("Accuracy")
    axs[1].set_ylim(bottom=0, top=1)
    axs[1].legend()
    axs[1].grid(visible=True, which="both", linewidth=0.5, alpha=0.5)
    axs[1].minorticks_on()
    axs[1].set_xlabel("Rounds")

    plt.tight_layout()
    plt.savefig(Path(save_path) / Path("loss_accuracy.png"), dpi=300)
    plt.close()


def plot_histories_with_accuracy(axs, histories, label):
    """Plot histories' accuracy and loss."""
    accuracies_across_runs = []
    losses_across_runs = []

    for run in histories:
        history_metrics = run.metrics_centralized["accuracy"]
        history_losses = run.losses_centralized

        rounds, accuracy_values = zip(*history_metrics)
        _, loss_values = zip(*history_losses)
        accuracies_across_runs.append(accuracy_values)
        losses_across_runs.append(loss_values)

    plot_accuracy_and_loss(
        axs, accuracies_across_runs, losses_across_runs, rounds, label
    )


def plot_accuracy_and_loss(
    axs, accuracies_across_runs, losses_across_runs, rounds, label
):
    """Plot histories' accuracy and loss."""
    x_axis = list(rounds)
    lowest_loss_across_runs = np.min(losses_across_runs, axis=0)
    highest_loss_across_runs = np.max(losses_across_runs, axis=0)
    lowest_accuracy_across_runs = np.min(accuracies_across_runs, axis=0)
    highest_accuracy_across_runs = np.max(accuracies_across_runs, axis=0)
    avg_loss_across_runs = np.add(lowest_loss_across_runs, highest_loss_across_runs) / 2
    avg_accuracy_across_runs = (
        np.add(lowest_accuracy_across_runs, highest_accuracy_across_runs) / 2
    )
    axs[0].fill_between(
        x_axis,
        highest_loss_across_runs,
        lowest_loss_across_runs,
        alpha=0.4,
        label=label,
    )
    axs[0].plot(x_axis, avg_loss_across_runs, linewidth=2)
    axs[1].fill_between(
        x_axis,
        highest_accuracy_across_runs,
        lowest_accuracy_across_runs,
        alpha=0.4,
        label=label,
    )
    axs[1].plot(x_axis, avg_accuracy_across_runs, linewidth=2)
