"""Utility functions for FedPara."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr.common import NDArrays
from flwr.server import History
from omegaconf import DictConfig
from torch.nn import Module


def plot_metric_from_history(
    hist: History,
    save_plot_path: str,
    model_size: float,
    cfg: DictConfig,
    suffix: str = "",
) -> None:
    """Plot the metrics from the history of the server.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : str
        Folder to save the plot to.
    model_size : float
        Size of the model in MB.
    cfg : Optional
        Optional dictionary containing the configuration of the experiment.
    suffix: Optional
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    _, axs = plt.subplots()
    rounds, values_accuracy = zip(*metric_dict["accuracy"])
    r_cc = (i * 2 * model_size * int(cfg.clients_per_round) / 1024 for i in rounds)

    # Set the title
    title = f"{cfg.strategy.algorithm} | parameters: {cfg.model.conv_type} | "
    title += (
        f"{cfg.dataset_config.name} {cfg.dataset_config.partition} | Seed {cfg.seed}"
    )
    axs.set_title(title)
    axs.grid(True)
    axs.plot(np.asarray([*r_cc]), np.asarray(values_accuracy))
    axs.set_ylabel("Accuracy")
    axs.set_xlabel("Communication Cost (GB)")
    fig_name = "_".join([metric_type, "metrics", suffix]) + ".png"
    plt.savefig(Path(save_plot_path) / Path(fig_name))
    plt.close()


def seed_everything(seed):
    """Seed everything for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_parameters(net: Module) -> NDArrays:
    """Get the parameters of the network."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
