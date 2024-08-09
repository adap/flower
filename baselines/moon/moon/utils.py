"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python modules. For
example, you may define here things like: loading a model from a checkpoint, saving
results, plotting.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from flwr.server.history import History


def compute_accuracy(model, dataloader, device="cpu", multiloader=False):
    """Compute accuracy."""
    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    loss_collector = []
    if multiloader:
        for loader in dataloader:
            with torch.no_grad():
                for batch in loader:
                    x = batch["img"]
                    target = batch["label"]
                    x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                    _, _, out = model(x)
                    if len(target) == 1:
                        loss = criterion(out, target)
                    else:
                        loss = criterion(out, target)
                    _, pred_label = torch.max(out.data, 1)
                    loss_collector.append(loss.item())
                    total += x.data.size()[0]
                    correct += (pred_label == target.data).sum().item()

                    if device == "cpu":
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.numpy()
                        )
                    else:
                        pred_labels_list = np.append(
                            pred_labels_list, pred_label.cpu().numpy()
                        )
                        true_labels_list = np.append(
                            true_labels_list, target.data.cpu().numpy()
                        )
        avg_loss = sum(loss_collector) / len(loss_collector)
    else:
        with torch.no_grad():
            for batch in dataloader:
                x = batch["img"]
                target = batch["label"]
                x, target = x.to(device), target.to(dtype=torch.int64).to(device)
                _, _, out = model(x)
                loss = criterion(out, target)
                _, pred_label = torch.max(out.data, 1)
                loss_collector.append(loss.item())
                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(
                        pred_labels_list, pred_label.cpu().numpy()
                    )
                    true_labels_list = np.append(
                        true_labels_list, target.data.cpu().numpy()
                    )
            avg_loss = sum(loss_collector) / len(loss_collector)

    if was_training:
        model.train()

    return correct / float(total), avg_loss


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Plot data from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.
    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    metric_type = "centralized"
    metric_dict = (
        hist.metrics_centralized
        if metric_type == "centralized"
        else hist.metrics_distributed
    )
    rounds, values = zip(*metric_dict["accuracy"])

    # Plot the curve
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, values)
    plt.xlabel("#round")
    plt.ylabel("Test accuracy")
    plt.legend()
    plt.show()

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()
