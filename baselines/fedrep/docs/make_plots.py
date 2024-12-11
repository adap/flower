"""Generate plots from json files."""

import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt


def read_from_results(path) -> Tuple[str, str, List]:
    """Load the json file with recorded configurations and results."""
    with open(path, "r") as fin:
        data = json.load(fin)
        algorithm = data["run_config"]["algorithm"]
        model = data["run_config"]["model-name"]
        accuracies = [res["accuracy"] * 100 for res in data["round_res"]]
        return algorithm, model, accuracies


def make_plot(dir_path, plt_title) -> None:
    """Given a directory with json files, generated a plot using the provided title."""
    plt.figure()
    with os.scandir(dir_path) as files:
        for file in files:
            algo, m, acc = read_from_results(file)
            rounds = [i + 1 for i in range(len(acc))]
            print(f"Max accuracy ({algo}): {max(acc):.2f}")
            plt.plot(rounds, acc, label=algo)
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.title(plt_title)
    plt.grid()
    plt.legend()
    plt.savefig(plt_title)


if __name__ == "__main__":
    # Plot naming convention is "Domain (#clients, #classes-per-client)".
    dir_cifar10_2 = "../results/cifar10/classes_2"
    make_plot(dir_cifar10_2, plt_title="CIFAR-10 (100, 2)")
    dir_cifar10_5 = "../results/cifar10/classes_5"
    make_plot(dir_cifar10_5, plt_title="CIFAR-10 (100, 5)")
    dir_cifar100_5 = "../results/cifar100/classes_5"
    make_plot(dir_cifar100_5, plt_title="CIFAR-100 (100, 5)")
    dir_cifar100_20 = "../results/cifar100/classes_20"
    make_plot(dir_cifar100_20, plt_title="CIFAR-100 (100, 20)")
