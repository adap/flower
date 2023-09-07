"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import random
from pathlib import Path
from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from flwr.common import NDArrays
from flwr.server import History
from torch.nn import Module


def plot_metric_from_history(
        hist: History,
        save_plot_path: Path,
        server_steps: List[float],
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
    server_steps: List[float]
        Server Learning rate across the training rounds.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])

    # rounds_loss, values_loss = zip(*hist.losses_centralized)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds), np.asarray(values))
    axs[0].set_ylabel("Accuracy")
    axs[1].plot(np.asarray(rounds), np.asarray(server_steps))
    axs[1].set_ylabel("Server Step Size")

    plt.xlabel("Rounds")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_parameters(net: Module) -> NDArrays:
    """Returns the parameters of a neural network."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]
