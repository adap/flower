"""Contain utility functions."""

import os
import pickle
import subprocess as sp
from secrets import token_hex
from typing import Dict, Optional

import psutil
from flwr.server.history import History


def dic_save(dictionary: Dict, filename: str):
    """Save a dictionary to file.

    Parameters
    ----------
    dictionary :
        Dictionary to be saves.
    filename : str
        Path to save the dictionary to.
    """
    with open(filename + ".pickle", "wb") as f:
        pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)


def dic_load(filename: str):
    """Load a dictionary from file.

    Parameters
    ----------
    filename : str
        Path to load the dictionary from.
    """
    try:
        with open(filename, "rb") as fp:
            return pickle.load(fp)
    except IOError:
        return {"checkpoint_round": 0}


def save_results_as_pickle(
    history: History,
    file_path: Optional[str],
) -> None:
    """Save results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store history.
    """

    def _add_random_suffix(file_name: str):
        """Add a randomly generated suffix to the file name."""
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return file_name + "_" + suffix + ".pkl"

    filename = "results.pkl"
    file_path = os.path.join(file_path, filename)

    if os.path.isfile(file_path):
        filename = _add_random_suffix("results")
        file_path = os.path.join(file_path, filename)

    print(f"Results will be saved into: {file_path}")

    data = {"history": history}

    # save results to pickle
    with open(file_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_gpu_memory():
    """Return gpu free memory."""
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)][0]
    memory_percent = (memory_free_values / 24564) * 100
    print(
        f"[Memory monitoring] Free memory GPU "
        f"{memory_free_values} MB, {memory_percent} %."
    )
    return memory_free_values


def get_cpu_memory():
    """Return cpu free memory."""
    # you can convert that object to a dictionary
    memory_info = psutil.virtual_memory()
    # you can have the percentage of used RAM
    memory_percent = 100.0 - memory_info.percent
    memory_free_values = memory_info.available / (1024 * 1024)  # in MB

    print(
        f"[Memory monitoring] Free memory CPU "
        f"{memory_free_values} MB, {memory_percent} %."
    )
    # you can calculate percentage of available memory
    return memory_free_values
