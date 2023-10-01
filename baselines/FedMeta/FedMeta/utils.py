"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from typing import List, Dict
import pickle
import os
import matplotlib.pyplot as plt

# Encoding list for the Shakespeare dataset
ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"


def _one_hot(
        index: int,
        size: int,
) -> List:
    """
    returns one-hot vector with given size and value 1 at given index

    """

    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(
        letter: str,
) -> int:
    """
    returns one-hot representation of given letter

    """

    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(
        word: str,
) -> List:
    """
    returns a list of character indices
    Args:
        word: string

    Return:
        indices: int list with length len(word)

    """

    indices = []
    for c in word:
        indices.append(ALL_LETTERS.find(c))
    return indices


def update_ema(
        prev_ema: float,
        current_value: float,
        smoothing_weight: float,
) -> float:
    """
    We use EMA to visually enhance the learning trend for each round.

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
    """
    Save parameters to visualize experiment results (Loss, ACC)

    Parameters
    ----------
    data_info : Dict
        This is a parameter dictionary of data from which the experiment was completed.
    """

    if os.path.exists(f"{data_info['path']}/{data_info['algo']}.pkl"):
        raise ValueError(f"'{data_info['path']}/{data_info['algo']}.pkl' is already exists!")

    with open(f"{data_info['path']}/{data_info['algo']}.pkl", 'wb') as f:
        pickle.dump(data_info, f)


def plot_from_pkl(directory="."):
    """
    Visualization of algorithms for each data (FedAvg, FedAvg_Meta, FedMeta_MAML, FedMeta_Meta-SGD)

    Parameters
    ----------
    directory : str
        Graph params directory path for Femnist or Shakespeare

    """

    color_mapping = {
        "fedavg.pkl": "green",
        "fedavg_meta.pkl": "blue",
        "fedmeta_maml.pkl": "orange",
        "fedmeta_meta_sgd.pkl": "red",
        # ... 여기에 추가 파일 이름과 색상을 매핑 ...
    }

    pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]

    all_data = {}

    for file in pkl_files:
        with open(os.path.join(directory, file), 'rb') as f:
            data = pickle.load(f)
            all_data[file] = data

    plt.figure(figsize=(14, 6))

    # Acc graph
    plt.subplot(1, 2, 1)
    for file, data in all_data.items():
        accuracies = [acc for _, acc in data["accuracy"]['accuracy']]
        plt.plot(accuracies, label=file, color=color_mapping.get(file, "black"))
    plt.title("Accuracy")
    plt.legend()

    # Loss graph
    plt.subplot(1, 2, 2)
    for file, data in all_data.items():
        loss = [loss for _, loss in data["loss"]]
        plt.plot(loss, label=file, color=color_mapping.get(file, "black"))
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()

    save_path = f"{directory}/result_graph.png"
    plt.savefig(save_path)

    plt.show()

if __name__ == '__main__':
    plot_from_pkl()
