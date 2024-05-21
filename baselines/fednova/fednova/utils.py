"""Utility functions for FedNova such as computing accuracy, plotting results, etc."""

import glob
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from flwr.common import Metrics
from omegaconf import DictConfig


def comp_accuracy(output, target, topk=(1,)):
    """Compute accuracy over the k top predictions wrt the target."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Return weighted average of accuracy metrics."""
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": np.sum(accuracies) / np.sum(examples)}


def fit_config(exp_config: DictConfig, server_round: int):
    """Return training configuration dict for each round.

    Learning rate is reduced by a factor after set rounds.
    """
    config = {}

    lr = exp_config.optimizer.lr

    if exp_config.lr_scheduling:
        if server_round == int(exp_config.num_rounds / 2):
            lr = exp_config.optimizer.lr / 10

        elif server_round == int(exp_config.num_rounds * 0.75):
            lr = exp_config.optimizer.lr / 100

    config["lr"] = lr
    config["server_round"] = server_round
    return config


# pylint: disable=too-many-locals, too-many-statements
def generate_plots(
    local_solvers: List[str], strategy: List[str], var_epochs: bool, momentum_plot=False
):
    """Generate plots for all experiments, saved in directory _static."""
    root_path = "multirun/"
    save_path = "_static/"

    def load_exp(exp_name: str, strat: str, var_epoch: bool):
        exp_dirs = os.path.join(
            root_path,
            f"optimizer_{exp_name.lower()}_strategy_"
            f"{strat.lower()}_var_local_epochs_{var_epoch}",
        )
        exp_files = glob.glob(f"{exp_dirs}/*/*.csv")

        exp_df = [pd.read_csv(f) for f in exp_files]
        exp_df = [df for df in exp_df if not df.isna().any().any()]

        assert len(exp_df) >= 1, (
            f"Atleast one results file must contain non-NaN values. "
            f"NaN values found in all seed runs of {exp_df}"
        )
        return exp_df

    def get_confidence_interval(data):
        """Return 95% confidence intervals along with mean."""
        avg = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        lower = avg - 1.96 * std / np.sqrt(len(data))
        upper = avg + 1.96 * std / np.sqrt(len(data))
        return avg, lower, upper

    # create tuple combination of experiment configuration for plotting
    # [("vanilla", "fedavg", True), ("vanilla", "fednova", True)]
    max_exp_len = max([len(local_solvers), len(strategy)])
    optim_exp_len = int(max_exp_len / len(local_solvers))
    strategy_exp_len = int(max_exp_len / len(strategy))
    var_epochs_len = int(max_exp_len)
    exp_list = list(
        zip(
            local_solvers * optim_exp_len,
            strategy * strategy_exp_len,
            [var_epochs] * var_epochs_len,
        )
    )

    exp_data = [load_exp(*args) for args in exp_list]

    # Iterate over each experiment
    plt.figure()
    title = ""
    for i, data_dfs in enumerate(exp_data):
        # Iterate over multiple seeds of same experiment
        combined_data = np.array([df["test_accuracy"].values for df in data_dfs])

        mean, lower_ci, upper_ci = get_confidence_interval(combined_data)

        epochs = np.arange(1, len(mean) + 1)

        optimizer, server_strategy, variable_epoch = exp_list[i]

        # Assign more readable legends for each plot according to paper
        if optimizer == "proximal" and server_strategy == "FedAvg":
            label = "FedProx"
        elif optimizer.lower() in ["server", "hybrid"]:
            label = optimizer
        elif optimizer.lower() == "vanilla" and momentum_plot:
            label = "No Momentum"
        else:
            label = server_strategy

        plt.plot(epochs, mean, label=label)
        plt.fill_between(epochs, lower_ci, upper_ci, alpha=0.3)

        if optimizer == "momentum":
            optimizer_label = "SGD-M"
        elif optimizer == "proximal":
            optimizer_label = "SGD w/ Proximal"
        else:
            optimizer_label = "SGD"

        if var_epochs:
            title = f"Local Solver: {optimizer_label}, Epochs ~ U(2, 5)"
        else:
            title = f"Local Solver: {optimizer_label}, Epochs = 2"

        print(
            f"---------------------Local Solver: {optimizer.upper()}, "
            f"Strategy: {server_strategy.upper()} Local Epochs Fixed: {variable_epoch}"
            f"---------------------"
        )
        print(f"Number of valid(not NaN) seeds for this experiment: {len(data_dfs)}")

        print(f"Test Accuracy: {mean[-1]:.2f} ± {upper_ci[-1] - mean[-1]:.2f}")

    if momentum_plot:
        title = "Comparison of Momentum Schemes: FedNova"
        save_name = "momentum_plot"
    else:
        save_name = local_solvers[0]

    plt.ylabel("Test Accuracy %", fontsize=12)
    plt.xlabel("Communication rounds", fontsize=12)
    plt.xlim([0, 103])
    plt.ylim([30, 80])
    plt.legend(loc="lower right", fontsize=12)
    plt.grid()
    plt.title(title, fontsize=15)
    plt.savefig(f"{save_path}testAccuracy_{save_name}_varEpochs_{var_epochs}.png")

    plt.show()


if __name__ == "__main__":
    for type_epoch_exp in [False, True]:
        for solver in ["vanilla", "momentum", "proximal"]:
            generate_plots([solver], ["FedAvg", "FedNova"], type_epoch_exp)

    generate_plots(
        ["Hybrid", "Server", "Vanilla"], ["FedNova"], True, momentum_plot=True
    )
