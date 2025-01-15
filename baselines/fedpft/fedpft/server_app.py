"""fedpft: A Flower Baseline."""

from typing import Callable, Dict, List, Tuple

from flwr.common import Context, Metrics, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from torch import device

from fedpft.strategy import FedPFT


def fedpft_get_on_fit_config_fn(
    n_mixtures: int, cov_type: str, seed: int, tol: float, max_iter: int
) -> Callable[[int], Dict[str, Scalar]]:
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
    def fit_config(server_round: int) -> Dict[str, Scalar]:
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
) -> Callable[[int], Dict[str, Scalar]]:
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
    def fit_config(server_round: int) -> Dict[str, Scalar]:
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


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Read from config
    num_rounds = context.run_config["num-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_evaluate = context.run_config["fraction-evaluate"]

    if context.run_config["strategy"] == "fedpft":
        strategy = FedPFT(
            num_classes=int(context.run_config["num-classes"]),
            feature_dimension=int(context.run_config["hidden-dimension"]),
            lr=float(context.run_config["server-lr"]),
            server_batch_size=int(context.run_config["server-batch-size"]),
            num_epochs=int(context.run_config["num-epochs"]),
            device=device(str(context.run_config["device"])),
            fraction_fit=float(fraction_fit),
            fraction_evaluate=float(fraction_evaluate),
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fedpft_get_on_fit_config_fn(
                n_mixtures=int(context.run_config["n-mixtures"]),
                cov_type=str(context.run_config["cov-type"]),
                seed=int(context.run_config["seed"]),
                tol=float(context.run_config["tol"]),
                max_iter=int(context.run_config["max-iter"]),
            ),
            accept_failures=bool(context.run_config["accept-failures"]),
        )
    else:
        strategy = FedAvg(
            fraction_fit=float(fraction_fit),
            fraction_evaluate=float(fraction_evaluate),
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fedavg_get_on_fit_config_fn(
                float(context.run_config["server-lr"]),
                int(context.run_config["num-epochs"]),
            ),
            accept_failures=bool(context.run_config["accept-failures"]),
        )

    config = ServerConfig(num_rounds=int(num_rounds))

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
