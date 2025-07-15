"""Utility functions for FedPara."""

import os
import pickle
import random
import time
from pathlib import Path
from secrets import token_hex
from typing import List, Optional

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
    metric_dict = (
        hist.metrics_centralized
        if hist.metrics_centralized
        else hist.metrics_distributed
    )
    _, axs = plt.subplots()
    rounds, values_accuracy = zip(*metric_dict["accuracy"])
    r_cc = (i * 2 * model_size * int(cfg.clients_per_round) / 1024 for i in rounds)

    # Set the title
    # make the suffix space seperated not underscore seperated
    title = " ".join(suffix.split("_"))
    axs.set_title(title)
    axs.grid(True)
    axs.plot(np.asarray([*r_cc]), np.asarray(values_accuracy))
    axs.set_ylabel("Accuracy")
    axs.set_xlabel("Communication Cost (GB)")
    fig_name = suffix + ".png"
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


def save_results_as_pickle(
    history: History,
    file_path: str,
    default_filename: Optional[str] = "history.pkl",
) -> None:
    """Save results from simulation to pickle.

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
        """Add a random suffix to the file name."""
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
        print("Using default filename")
        if default_filename is None:
            return path_
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")
    # data = {"history": history, **extra_results}
    data = {"history": history}
    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def set_client_state_save_path(path: str) -> str:
    """Set the client state save path."""
    client_state_save_path = time.strftime("%Y-%m-%d")
    client_state_sub_path = time.strftime("%H-%M-%S")
    client_state_save_path = f"{path}{client_state_save_path}/{client_state_sub_path}"
    if not os.path.exists(client_state_save_path):
        os.makedirs(client_state_save_path)
    return client_state_save_path


def get_keys_state_dict(model, algorithm, mode: str = "local") -> List[str]:
    """."""
    keys: List[str] = []
    match algorithm:
        case "fedper":
            if mode == "local":
                keys = list(filter(lambda x: "fc1" not in x, model.state_dict().keys()))
            elif mode == "global":
                keys = list(filter(lambda x: "fc1" in x, model.state_dict().keys()))
        case "pfedpara":
            if mode == "local":
                keys = list(filter(lambda x: "w2" in x, model.state_dict().keys()))
            elif mode == "global":
                keys = list(filter(lambda x: "w1" in x, model.state_dict().keys()))
        case _:
            raise NotImplementedError(f"algorithm {algorithm} not implemented")

    return keys
