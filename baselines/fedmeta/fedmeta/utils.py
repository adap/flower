"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt

# Encoding list for the Shakespeare dataset
ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


def _one_hot(
    index: int,
    size: int,
) -> List:
    """Return one-hot vector with given size and value 1 at given index."""
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(
    letter: str,
) -> int:
    """Return one-hot representation of given letter."""
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(
    word: str,
) -> List:
    """Return a list of character indices.

    Parameters
    ----------
        word: string.

    Returns
    -------
        indices: int list with length len(word)
    """
    indices = []
    for count in word:
        indices.append(ALL_LETTERS.find(count))
    return indices


def update_ema(
    prev_ema: float,
    current_value: float,
    smoothing_weight: float,
) -> float:
    """We use EMA to visually enhance the learning trend for each round.

    Parameters
    ----------
    prev_ema : float
        The list of metrics to aggregate.
    current_value : float
        The list of metrics to aggregate.
    smoothing_weight : float
        The list of metrics to aggregate.


    Returns
    -------
    EMA_Loss or EMA_ACC
        The weighted average metric.
    """
    if prev_ema is None:
        return current_value
    return (1 - smoothing_weight) * current_value + smoothing_weight * prev_ema


def save_graph_params(data_info: Dict):
    """Save parameters to visualize experiment results (Loss, ACC).

    Parameters
    ----------
    data_info : Dict
        This is a parameter dictionary of data from which the experiment was completed.
    """
    if os.path.exists(f"{data_info['path']}/{data_info['algo']}.pkl"):
        raise ValueError(
            f"'{data_info['path']}/{data_info['algo']}.pkl' is already exists!"
        )

    with open(f"{data_info['path']}/{data_info['algo']}.pkl", "wb") as file:
        pickle.dump(data_info, file)


def plot_from_pkl(directory="."):
    """Visualization of algorithms like 4 Algorithm for data.

    Parameters
    ----------
    directory : str
        Graph params directory path for Femnist or Shakespeare
    """
    color_mapping = {
        "fedavg.pkl": "#66CC00",
        "fedavg_meta.pkl": "#3333CC",
        "fedmeta_maml.pkl": "#FFCC00",
        "fedmeta_meta_sgd.pkl": "#CC0000",
    }

    pkl_files = [f for f in os.listdir(directory) if f.endswith(".pkl")]

    all_data = {}

    for file in pkl_files:
        with open(os.path.join(directory, file), "rb") as file_:
            data = pickle.load(file_)
            all_data[file] = data

    plt.figure(figsize=(7, 12))

    # Acc graph
    plt.subplot(2, 1, 1)
    for file in sorted(all_data.keys()):
        data = all_data[file]
        accuracies = [acc for _, acc in data["accuracy"]["accuracy"]]
        legend_ = file[:-4] if file.endswith(".pkl") else file
        plt.plot(
            accuracies,
            label=legend_,
            color=color_mapping.get(file, "black"),
            linewidth=3,
        )
    plt.title("Accuracy")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    for file in sorted(all_data.keys()):
        data = all_data[file]
        loss = [loss for _, loss in data["loss"]]
        legend_ = file[:-4] if file.endswith(".pkl") else file
        plt.plot(
            loss, label=legend_, color=color_mapping.get(file, "black"), linewidth=3
        )
    plt.title("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    save_path = f"{directory}/result_graph.png"
    plt.savefig(save_path)

    plt.show()
