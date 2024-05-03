"""Contains utility functions."""

import errno
import os
from pathlib import Path

import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig


def preprocess_input(cfg_model, cfg_data):
    """Preprocess the input to get input shape, other derivables.

    Parameters
    ----------
    cfg_model : DictConfig
        Retrieve model-related information from the base.yaml configuration in Hydra.
    cfg_data : DictConfig
        Retrieve data-related information required to construct the model.

    Returns
    -------
    Dict
        Dictionary contained derived information from config.
    """
    model_config = {}
    # if cfg_model.model_name == "conv":
    #     model_config["model_name"] =
    # elif for others...
    model_config["model"] = cfg_model.model_name
    if cfg_data.dataset_name == "MNIST":
        model_config["data_shape"] = [1, 28, 28]
        model_config["classes_size"] = 10
    elif cfg_data.dataset_name == "CIFAR10":
        model_config["data_shape"] = [3, 32, 32]
        model_config["classes_size"] = 10

    if "hidden_layers" in cfg_model:
        model_config["hidden_layers"] = cfg_model.hidden_layers
    if "norm" in cfg_model:
        model_config["norm"] = cfg_model.norm
    if "scale" in cfg_model:
        model_config["scale"] = cfg_model.scale
    if "mask" in cfg_model:
        model_config["mask"] = cfg_model.mask

    return model_config


def make_optimizer(optimizer_name, parameters, learning_rate, weight_decay, momentum):
    """Make the optimizer with given config.

    Parameters
    ----------
    optimizer_name : str
        Name of the optimizer.
    parameters : Dict
        Parameters of the model.
    learning_rate: float
        Learning rate of the optimizer.
    weight_decay: float
        weight_decay of the optimizer.

    Returns
    -------
    torch.optim.Optimizer
        Optimizer.
    """
    optimizer = None
    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay
        )
    return optimizer


def make_scheduler(scheduler_name, optimizer, milestones):
    """Make the scheduler with given config.

    Parameters
    ----------
    scheduler_name : str
        Name of the scheduler.
    optimizer : torch.optim.Optimizer
        Parameters of the model.
    milestones: List[int]
        List of epoch indices. Must be increasing.

    Returns
    -------
    torch.optim.lr_scheduler.Scheduler
        scheduler.
    """
    scheduler = None
    if scheduler_name == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones
        )
    return scheduler


def get_global_model_rate(model_mode):
    """Give the global model rate from string(cfg.control.model_mode) .

    Parameters
    ----------
    model_mode : str
        Contains the division of computational complexties among clients.

    Returns
    -------
    str
        global model computational complexity.
    """
    model_mode = "" + model_mode
    model_mode = model_mode.split("-")[0][0]
    return model_mode


class ModelRateManager:
    """Control the model rate of clients in case of simulation."""

    def __init__(self, model_split_mode, model_split_rate, model_mode):
        self.model_split_mode = model_split_mode
        self.model_split_rate = model_split_rate
        self.model_mode = model_mode
        self.model_mode = self.model_mode.split("-")

    def create_model_rate_mapping(self, num_users):
        """Change the client to model rate mapping accordingly."""
        client_model_rate = []

        if self.model_split_mode == "fix":
            mode_rate, proportion = [], []
            for comp_level_prop in self.model_mode:
                mode_rate.append(self.model_split_rate[comp_level_prop[0]])
                proportion.append(int(comp_level_prop[1:]))
            num_users_proportion = num_users // sum(proportion)
            for i, comp_level in enumerate(mode_rate):
                client_model_rate += np.repeat(
                    comp_level, num_users_proportion * proportion[i]
                ).tolist()
            client_model_rate = client_model_rate + [
                client_model_rate[-1] for _ in range(num_users - len(client_model_rate))
            ]
            # return client_model_rate

        elif self.model_split_mode == "dynamic":
            mode_rate, proportion = [], []

            for comp_level_prop in self.model_mode:
                mode_rate.append(self.model_split_rate[comp_level_prop[0]])
                proportion.append(int(comp_level_prop[1:]))

            proportion = (np.array(proportion) / sum(proportion)).tolist()

            rate_idx = torch.multinomial(
                torch.tensor(proportion), num_samples=num_users, replacement=True
            ).tolist()
            client_model_rate = np.array(mode_rate)[rate_idx]

            # return client_model_rate

        else:
            raise ValueError("Not valid model split mode")

        return client_model_rate


def save_model(model, path):
    """To save the model in the given path."""
    # print('in save model')
    current_path = HydraConfig.get().runtime.output_dir
    model_save_path = Path(current_path) / path
    torch.save(model.state_dict(), model_save_path)


# """ The following functions(check_exists, makedir_exit_ok, save, load)
# are adopted from authors (of heterofl) implementation."""


def check_exists(path):
    """Check if the given path exists."""
    return os.path.exists(path)


def makedir_exist_ok(path):
    """Create a directory."""
    try:
        os.makedirs(path)
    except OSError as os_err:
        if os_err.errno == errno.EEXIST:
            pass
        else:
            raise


def save(inp, path, protocol=2, mode="torch"):
    """Save the inp in a given path."""
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == "torch":
        torch.save(inp, path, pickle_protocol=protocol)
    elif mode == "numpy":
        np.save(path, inp, allow_pickle=True)
    else:
        raise ValueError("Not valid save mode")


# pylint: disable=no-else-return
def load(path, mode="torch"):
    """Load the file from given path."""
    if mode == "torch":
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == "numpy":
        return np.load(path, allow_pickle=True)
    else:
        raise ValueError("Not valid save mode")
