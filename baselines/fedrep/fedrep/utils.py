"""Utility functions for FedRep."""

import os
import pickle
import time
from pathlib import Path
from secrets import token_hex
from typing import Callable, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History
from omegaconf import DictConfig

from fedrep.client import BaseClient, FedRepClient, get_client_fn_simulation
from fedrep.implemented_models.cnn_cifar10 import CNNCifar10, CNNCifar10ModelSplit
from fedrep.implemented_models.cnn_cifar100 import CNNCifar100, CNNCifar100ModelSplit


def set_model_class(config: DictConfig) -> DictConfig:
    """Set model class based on the model name in the config file."""
    # Set the model class
    if config.dataset.name.lower() == "cifar10":
        config.model["_target_"] = "fedrep.implemented_models.cnn_cifar10.CNNCifar10"
    elif config.dataset.name.lower() == "cifar100":
        config.model["_target_"] = "fedrep.implemented_models.cnn_cifar100.CNNCifar100"
    else:
        raise NotImplementedError(f"Model for {config.dataset.name} not implemented")
    return config


def set_server_target(config: DictConfig) -> DictConfig:
    """Set the server target based on the algorithm in the config file."""
    # Set the server target
    if config.algorithm.lower() == "fedrep":
        config.strategy["_target_"] = "fedrep.server.AggregateBodyStrategyPipeline"
    elif config.algorithm.lower() == "fedavg":
        config.strategy["_target_"] = "fedrep.server.DefaultStrategyPipeline"
    else:
        raise NotImplementedError(f"Algorithm {config.algorithm} not implemented")
    return config


def set_client_state_save_path() -> str:
    """Set the client state save path."""
    client_state_save_path = time.strftime("%Y-%m-%d")
    client_state_sub_path = time.strftime("%H-%M-%S")
    client_state_save_path = (
        f"./client_states/{client_state_save_path}/{client_state_sub_path}"
    )
    if not os.path.exists(client_state_save_path):
        os.makedirs(client_state_save_path)
    return client_state_save_path


def get_client_fn(
    config: DictConfig, client_state_save_path: str = ""
) -> Callable[[str], Union[FedRepClient, BaseClient]]:
    """Get client function."""
    # Get algorithm
    algorithm = config.algorithm.lower()
    # Get client fn
    if algorithm == "fedrep":
        client_fn = get_client_fn_simulation(
            config=config, client_state_save_path=client_state_save_path
        )
    elif algorithm == "fedavg":
        client_fn = get_client_fn_simulation(config=config)
    else:
        raise NotImplementedError
    return client_fn


def get_create_model_fn(
    config: DictConfig,
) -> tuple[
    Callable[[], Union[type[CNNCifar10], type[CNNCifar100]]],
    Union[type[CNNCifar10ModelSplit], type[CNNCifar100ModelSplit]],
]:
    """Get create model function."""
    device = config.server_device
    if config.model_name.lower() == "cnncifar10":
        split = CNNCifar10ModelSplit

        def create_cnncifar10() -> Type[CNNCifar10]:
            """Create initial CNNCifar10 model."""
            return CNNCifar10().to(device)

        return create_cnncifar10, split

    if config.model_name.lower() == "cnncifar100":
        split = CNNCifar100ModelSplit

        def create_cnncifar100() -> Type[CNNCifar100]:
            """Create initial CNNCifar100 model."""
            return CNNCifar100().to(device)

        return create_cnncifar100, split

    raise NotImplementedError("Model not implemented. Check name.")


def plot_metric_from_history(
    hist: History, save_plot_path: Path, suffix: Optional[str] = ""
) -> None:
    """Plot from Flower server History.

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
    _, values = zip(*metric_dict["accuracy"])

    rounds_loss, values_loss = zip(*hist.losses_distributed)

    _, axs = plt.subplots(nrows=2, ncols=1, sharex="row")  # type: ignore
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))  # type: ignore
    axs[1].plot(np.asarray(rounds_loss), np.asarray(values))  # type: ignore

    axs[0].set_ylabel("Loss")  # type: ignore
    axs[1].set_ylabel("Accuracy")  # type: ignore

    axs[0].grid()  # type: ignore
    axs[1].grid()  # type: ignore
    plt.xlabel("Rounds")
    # plt.legend(loc="lower right")

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


def save_results_as_pickle(
    history: History,
    file_path: Union[str, Path],
    default_filename: Optional[str] = "results.pkl",
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
