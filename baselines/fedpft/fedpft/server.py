"""Create global evaluation function."""

from typing import Callable, Dict, List, Tuple

from flwr.common import Metrics


def fedpft_get_on_fit_config_fn(
    n_mixtures: int, cov_type: str, seed: int, tol: float, max_iter: int
) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns FedPFT training configurations.

    Parameters
    ----------
    n_mixtures : int
        Number of mixtures for GMMs
    cov_type : str
        Type of covariance
    seed : int
        Seed for learning and sampling from the GMMs
    tol : float
        Error tolerance for learning GMMs
    max_iter : int
        Maximum number of iteration for EM algorithm

    Returns
    -------
    Callable[[int], Dict[str, str]]
        Function to return a config with the `lr` and `num_epochs`
    """

    # pylint: disable=unused-argument
    def fit_config(server_round: int) -> Dict[str, str]:
        """Return a configuration for training Gaussian Mixtures."""
        config = {
            "n_mixtures": str(n_mixtures),
            "cov_type": cov_type,
            "seed": str(seed),
            "tol": str(tol),
            "max_iter": str(max_iter),
        }
        return config

    return fit_config


def fedavg_get_on_fit_config_fn(
    learning_rate: float,
    num_epochs: int,
) -> Callable[[int], Dict[str, str]]:
    """Return a function which returns FedAvg training configurations.

    Parameters
    ----------
    learning_rate : float
        Client's learning rate
    num_epochs : int
        Number of epochs for local learning of clients

    Returns
    -------
    Callable[[int], Dict[str, str]]
        Function to return a config with the `learning_rate` and `num_epochs`
    """

    # pylint: disable=unused-argument
    def fit_config(server_round: int) -> Dict[str, str]:
        """Return a configuration number of epochs and learning rate."""
        config = {
            "lr": str(learning_rate),
            "num_epochs": str(num_epochs),
        }
        return config

    return fit_config


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate with weighted average during evaluation.

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
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": int(sum(accuracies)) / int(sum(examples))}
