"""Utility functions for FedNova such as computing accuracy, plotting results, etc."""


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


def generate_plots(local_solver: str = "vanilla", var_epochs: bool = False):
    """Generate plots for the experiment."""
    metrics = ["test_accuracy"]
    base_path = "results/"
    save_path = "../_static/"
    all_files = os.listdir(base_path)

    if local_solver == "proximal":
        baseline_strategy = "fedprox"
    else:
        baseline_strategy = "fedavg"

    baseline_files = [
        os.path.join(base_path, f)
        for f in all_files
        if f.startswith(f"{local_solver}_fedavg_varEpoch_{var_epochs}_")
    ]
    fednova_files = [
        os.path.join(base_path, f)
        for f in all_files
        if f.startswith(f"{local_solver}_fednova_varEpoch_{var_epochs}_")
    ]

    baseline_df = [pd.read_csv(f) for f in baseline_files]
    baseline_df = [df for df in baseline_df if not df.isna().any().any()]

    assert len(baseline_df) >= 1, (
        f"Atleast one results file must contain non-NaN values. "
        f"NaN values found in {baseline_files}"
    )

    fednova_df = [pd.read_csv(f) for f in fednova_files]
    fednova_df = [df for df in fednova_df if not df.isna().any().any()]

    assert len(fednova_df) >= 1, (
        f"Atleast one results file must contain non-NaN values. "
        f"NaN values found in {fednova_files}"
    )

    def get_confidence_interval(data):
        """Return 95% confidence intervals along with mean."""
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        lower = mean - 1.96 * std / np.sqrt(len(data))
        upper = mean + 1.96 * std / np.sqrt(len(data))
        return mean, lower, upper

    for metric in metrics:
        baseline_metric_data = np.array([df[metric].values for df in baseline_df])
        fednova_metric_data = np.array([df[metric].values for df in fednova_df])

        baseline_mean, baseline_lower, baseline_upper = get_confidence_interval(
            baseline_metric_data
        )
        fednova_mean, fednova_lower, fednova_upper = get_confidence_interval(
            fednova_metric_data
        )

        epochs = np.arange(1, len(baseline_mean) + 1)

        plt.figure()
        if baseline_strategy == "fedavg":
            baseline_label = "FedAvg"
        elif baseline_strategy == "fedprox":
            baseline_label = "FedProx"

        plt.plot(epochs, baseline_mean, label=baseline_label)
        plt.fill_between(epochs, baseline_lower, baseline_upper, alpha=0.3)
        plt.plot(epochs, fednova_mean, label="FedNova", c="red")
        plt.fill_between(epochs, fednova_lower, fednova_upper, alpha=0.3, color="red")
        plt.ylabel("Test Accuracy %")
        plt.xlabel("Communication rounds")
        plt.xlim([0, 103])
        plt.ylim([30, 80])
        plt.legend(loc="lower right")
        plt.grid()
        if local_solver == "momentum":
            optimizer = "SGD-M"
        elif local_solver == "proximal":
            optimizer = "SGD w/ Proximal"
        else:
            optimizer = "SGD"
        if var_epochs:
            title = f"Local Solver: {optimizer}, Epochs ~ U(2, 5)"
        else:
            title = f"Local Solver: {optimizer}, Epochs = 2"
        plt.title(title)
        plt.savefig(
            f"{save_path}testAccuracy_{local_solver}_varEpochs_{var_epochs}.png"
        )


        print(
            f"---------------------------Local Solver: {local_solver.upper()} "
            f"Var Epochs: {var_epochs}---------------------------"
        )

        print(
            "Number of valid seeds: Baseline: {} FedNova: {} ".format(len(baseline_df),
                                                                      len(fednova_df)))

        print(
            f"{baseline_label}: {baseline_mean[-1]:.2f} ± "
            f"{baseline_upper[-1] - baseline_mean[-1]:.2f}"
        )
        print(
            f"FedNova: {fednova_mean[-1]:.2f} ± "
            f"{fednova_upper[-1] - fednova_mean[-1]:.2f}"
        )

        plt.show()


if __name__ == "__main__":
    for var_epochs in [False, True]:
        for local_solver in ["vanilla", "momentum", "proximal"]:
            generate_plots(local_solver=local_solver, var_epochs=var_epochs)
