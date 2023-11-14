"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


##########################################################
# UTILS #
##########################################################
def set_seed(seed):
    """Set the seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def show_plots():
    """Display the plots based on results stored in res.json."""
    with open("res.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # Use the loaded data
    rounds = [entry["round"] for entry in data]
    loss_values = [entry["avg_loss"] for entry in data]
    accuracy_values = [entry["avg_accuracy"] for entry in data]

    plt.figure(figsize=(8, 4))
    plt.plot(rounds, loss_values, color="blue", label="Loss")
    plt.title("<>")
    plt.xlabel("Round")
    plt.ylabel("Loss")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(rounds, accuracy_values, color="green", label="Accuracy")
    plt.title("<>")
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.show()
