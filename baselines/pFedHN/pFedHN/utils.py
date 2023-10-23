"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import json
import random

import numpy as np
import torch
from matplotlib import pyplot as plt


##########################################################
# UTILS #
##########################################################
def set_seed(seed):
    """Set the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(no_cuda=False, gpus="0"):
    """Get the device."""
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )


##########################################################
# THE PLOTS #
##########################################################


def show_plots():
    """Display the plots based on results stored in res.json."""
    with open("res.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # Use the loaded data
    rounds = [entry["round"] for entry in data]
    loss_values = [entry["loss"] for entry in data]
    accuracy_values = [entry["accuracies"] for entry in data]

    plt.figure(figsize=(8, 4))
    plt.plot(rounds, loss_values, marker="o")
    plt.title("Loss vs Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Accuracy vs Round
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, accuracy_values, marker="o", color="green")
    plt.title("Accuracy vs Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # Accuracy vs Loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_values, accuracy_values, marker="o", color="red")
    plt.title("Accuracy vs Loss")
    plt.xlabel("Loss")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()
