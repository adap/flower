"""Contains utility functions for CNN FL on CIFAR10."""


from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from flwr.common import Metrics
from flwr.common.typing import NDArrays, Scalar
from flwr.server.history import History
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from flwr_baselines.publications.fedavg.cifar10 import model


def plot_metric_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

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

    _, axis = plt.subplots()
    plt.plot(np.asarray(rounds), np.asarray(values), c="b", label="FedAvg")

    # Set paper's results
    plt.axhline(
        y=max(values),
        color="r",
        linestyle="dashed",
        label="Paper's baseline accuracy ",
    )

    plt.title(f"{metric_type.capitalize()} Validation - CIFAR10")
    plt.xlabel("Rounds")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    # Set the apect ratio to 1.0
    xleft, xright = axis.get_xlim()
    ybottom, ytop = axis.get_ylim()
    axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)

    plt.savefig(Path(save_plot_path) / Path(f"{metric_type}_metrics{suffix}.png"))
    plt.close()


def plot_eval_loss_from_history(
    hist: History,
    save_plot_path: Path,
    suffix: Optional[str] = "",
) -> None:
    """Function to plot from Flower server History.

    Parameters
    ----------
    hist : History
        Object containing evaluation for all rounds.
    save_plot_path : Path
        Folder to save the plot to.

    suffix: Optional[str]
        Optional string to add at the end of the filename for the plot.
    """
    loss_type = "centralized"
    loss_dict = (
        hist.losses_centralized
        if loss_type == "centralized"
        else hist.losses_distributed
    )
    x = zip(*loss_dict)
    loss_rounds, eval_loss = list(x)

    _, axis = plt.subplots()
    plt.plot(np.asarray(loss_rounds), np.asarray(eval_loss), c="r", label="FedAvg")
    # Set paper's results
    plt.axhline(
        y=min(eval_loss),
        color="b",
        linestyle="dashed",
        label="Paper's baseline loss ",
    )

    plt.title(f"{loss_type.capitalize()} Validation - CIFAR 10")
    plt.xlabel("Rounds")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    xleft, xright = axis.get_xlim()
    ybottom, ytop = axis.get_ylim()
    axis.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * 1.0)

    plt.savefig(Path(save_plot_path) / Path(f"{loss_type}_loss{suffix}.png"))
    plt.close()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregation function for weighted average during evaluation.

    Parameters
    ----------
    metrics : List[Tuple[int, Metrics]]
        The list of metrics to aggregate.

    Returns
    -------
    Metrics
        The weighted average metric.
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "accuracy": int(sum(list(map(bool, accuracies))))
        / int(sum(list(map(bool, examples))))
    }


def gen_evaluate_fn(
    testloader: DataLoader, device: torch.device
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generates the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        """Use the entire MNIST test set for evaluation."""
        # determine device
        net = model.Net()
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        loss, accuracy = model.test(net, testloader, device=device)
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate
