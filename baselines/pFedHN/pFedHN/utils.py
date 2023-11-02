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


##########################################################
# THE PLOTS #
##########################################################


def show_plots():
    """Display the plots based on results stored in res.json."""
    with open("res_test_1.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # Use the loaded data
    rounds = [entry["round"] for entry in data]
    loss_values = [entry["loss"] for entry in data]
    accuracy_values = [entry["accuracies"] for entry in data]

    plt.figure(figsize=(8, 4))
    plt.plot(rounds, loss_values)
    plt.title("Loss vs Round")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.show()

    # Accuracy vs Round
    plt.figure(figsize=(8, 4))
    plt.plot(rounds, accuracy_values, color="green")
    plt.title("Accuracy vs Round")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.show()


# show_plots()
