"""Contain utility functions."""

import pickle
import subprocess as sp
from pathlib import Path
from secrets import token_hex
from typing import Dict, Optional, Union

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
    file_path: Union[str, Path],
    extra_results=None,
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
    if extra_results is None:
        extra_results = {}
    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Add a randomly generated suffix to the file name."""
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
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
