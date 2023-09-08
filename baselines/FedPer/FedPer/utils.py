"""Utility functions for FedPer."""
import os
import pickle
import time
from pathlib import Path
from secrets import token_hex
from typing import Callable, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from flwr.server.history import History
from omegaconf import DictConfig

from FedPer.client import BaseClient, FedPerClient, get_client_fn_simulation
from FedPer.implemented_models.cnn_model import CNNModelSplit, CNNNet
from FedPer.implemented_models.mobile_model import MobileNet, MobileNetModelSplit
from FedPer.implemented_models.resnet_model import ResNet, ResNetModelSplit


def set_model_class(config: DictConfig) -> DictConfig:
    """Set model class based on the model name in the config file."""
    # Set the model class
    if config.model.name.lower() == "resnet":
        config.model._target_ = "FedPer.implemented_models.resnet_model.ResNet"
    elif config.model.name.lower() == "mobile":
        config.model._target_ = "FedPer.implemented_models.mobile_model.MobileNet"
    elif config.model.name.lower() == "cnn":
        config.model._target_ = "FedPer.implemented_models.cnn_model.CNNNet"
    else:
        raise NotImplementedError(f"Model {config.model.name} not implemented")
    return config


def set_num_classes(config: DictConfig) -> DictConfig:
    """Set the number of classes based on the dataset name in the config file."""
    # Set the number of classes
    if config.dataset.name.lower() == "cifar10":
        config.dataset.num_classes = 10
    elif config.dataset.name.lower() == "cifar100":
        config.dataset.num_classes = 100
    elif config.dataset.name.lower() == "flickr":
        config.dataset.num_classes = 5
    else:
        raise NotImplementedError(f"Dataset {config.dataset.name} not implemented")
    return config


def set_server_target(config: DictConfig) -> DictConfig:
    """Set the server target based on the algorithm in the config file."""
    # Set the server target
    if config.algorithm.lower() == "fedper":
        config.strategy._target_ = "FedPer.server.AggregateBodyStrategyPipeline"
    elif config.algorithm.lower() == "fedavg":
        config.strategy._target_ = "FedPer.server.DefaultStrategyPipeline"
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
    config: DictConfig, client_state_save_path: str = None
) -> Callable[[str], Union[FedPerClient, BaseClient]]:
    """Get client function."""
    # Get algorithm
    algorithm = config.algorithm.lower()
    # Get client fn
    if algorithm == "fedper":
        client_fn = get_client_fn_simulation(
            config=config,
            client_state_save_path=client_state_save_path,
        )
    elif algorithm == "fedavg":
        client_fn = get_client_fn_simulation(
            config=config,
        )
    else:
        raise NotImplementedError
    return client_fn


def get_create_model_fn(
    config: DictConfig,
) -> Union[
    Tuple[Callable[[], CNNNet], CNNModelSplit],
    Tuple[Callable[[], MobileNet], MobileNetModelSplit],
    Tuple[Callable[[], ResNet], ResNetModelSplit],
]:
    """Get create model function."""
    device = config.server_device

    if config.model.name.lower() == "cnn":
        split = CNNModelSplit

        def create_model() -> CNNNet:
            """Create initial CNN model."""
            return CNNNet(name="cnn").to(device)

    elif config.model.name.lower() == "mobile":
        split = MobileNetModelSplit

        def create_model() -> MobileNet:
            """Create initial MobileNet-v1 model."""
            return MobileNet(
                num_head_layers=config.model.num_head_layers,
                num_classes=config.model.num_classes,
                name=config.model.name,
                device=config.model.device,
            ).to(device)

    elif config.model.name.lower() == "resnet":
        split = ResNetModelSplit

        def create_model() -> ResNet:
            """Create initial ResNet model."""
            return ResNet(
                num_head_layers=config.model.num_head_layers,
                num_classes=config.model.num_classes,
                name=config.model.name,
                device=config.model.device,
            ).to(device)

    else:
        raise NotImplementedError("Model not implemented, check name. ")
    return create_model, split


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
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
    rounds, values = zip(*metric_dict["accuracy"])

    # let's extract decentralized loss (main metric reported in FedProx paper)
    rounds_loss, values_loss = zip(*hist.losses_distributed)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex="row")
    axs[0].plot(np.asarray(rounds_loss), np.asarray(values_loss))
    axs[1].plot(np.asarray(rounds_loss), np.asarray(values))

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
    extra_results: Optional[Dict],
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
        else:
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
