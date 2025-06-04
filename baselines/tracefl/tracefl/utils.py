"""TraceFL Utilities Module.

This module provides utility functions for the TraceFL system, including configuration
management and experiment tracking.
"""

from typing import Any, Callable, List, Mapping, Optional, Union

import numpy as np
from sklearn import metrics


def safe_len(obj: Optional[Any]) -> int:
    """Safely compute the length if object is not None."""
    return len(obj) if obj is not None else 0


def get_label_count(
    client_label_dict: Mapping[Union[str, int], int],
    target_label: Any,
) -> int:
    """Return count of target_label from client_label_dict.

    Accepts dicts with str or int keys to support different label formats. Tries the key
    as-is, then as a string.
    """
    return client_label_dict.get(
        target_label, client_label_dict.get(str(target_label), 0)
    )


def safe_max(items: List[Any], scoring_fn: Callable[[Any], float]) -> Any:
    """Safe wrapper for max with a well-typed key function."""
    return max(items, key=scoring_fn)


def get_backend_config(cfg):
    """Get the backend configuration for Ray-based federated learning execution.

    This function is used by the FLSimulation class to configure Ray backend settings
    for distributed federated learning execution. It extracts client resource
    requirements from the configuration and formats them for Ray's distributed
    computing framework.

    Args:
        cfg: Configuration object containing client resource settings under
             cfg.tool.tracefl.client_resources (cpus, gpus)

    Returns
    -------
        dict: Dictionary containing backend configuration parameters including:
            - client_resources: CPU/GPU allocation per client
            - init_args: Ray initialization arguments
            - actor: Framework-specific actor configurations

    Note:
        This is specifically designed for Ray-based distributed FL execution.
        The returned configuration is used by FLSimulation when setting up
        the distributed computing environment for federated learning.
    """
    client_resources = {
        "num_cpus": cfg.tool.tracefl.client_resources.cpus,
        "num_gpus": cfg.tool.tracefl.client_resources.gpus,
    }
    return {
        "client_resources": client_resources,
        "init_args": {"log_to_driver": True, "logging_level": 30},
        "actor": {"tensorflow": 0},
    }


def compute_importance(n_elements: int, decay_factor: float = 0.9) -> List[float]:
    """Compute importance weights for a sequence of elements using a decay factor.

    Parameters
    ----------
    n_elements : int
        The number of elements for which importance is computed.
    decay_factor : float, optional
        The decay factor used to compute importance (default is 0.9).

    Returns
    -------
    list of float
        A list of importance weights for the elements.
    """
    indices = np.arange(n_elements)
    importance = decay_factor ** indices[::-1]
    return importance.tolist()


def set_exp_key(cfg):
    """Set the experiment key based on configuration parameters.

    Args:
        cfg: Configuration object containing experiment parameters

    Returns
    -------
        str: A unique key identifying the experiment configuration
    """
    # Check if DP is enabled
    dp_enabled = (
        cfg.tool.tracefl.strategy.noise_multiplier > 0
        and cfg.tool.tracefl.strategy.clipping_norm > 0
    )

    if dp_enabled:
        dp_key = (
            f"DP-(noise{cfg.tool.tracefl.strategy.noise_multiplier}+"
            f"clip{cfg.tool.tracefl.strategy.clipping_norm})-"
            f"{cfg.tool.tracefl.exp_key}-"
            f"{cfg.tool.tracefl.model.name}-{cfg.tool.tracefl.dataset.name}-"
            f"faulty_clients[[]]-"
            f"TClients{cfg.tool.tracefl.data_dist.num_clients}-"
            f"{cfg.tool.tracefl.strategy.name}-(R{cfg.tool.tracefl.strategy.num_rounds}"
            f"-clientsPerR{cfg.tool.tracefl.strategy.clients_per_round})"
            f"-{cfg.tool.tracefl.data_dist.dist_type}"
            f"{cfg.tool.tracefl.data_dist.dirichlet_alpha}"
            f"-batch{cfg.tool.tracefl.data_dist.batch_size}"
            f"-epochs{cfg.tool.tracefl.client.epochs}-"
            f"lr{cfg.tool.tracefl.client.lr}"
        )
        return dp_key

    # Non-DP experiment key
    non_dp_key = (
        f"NonDP-{cfg.tool.tracefl.exp_key}-"
        f"{cfg.tool.tracefl.model.name}-{cfg.tool.tracefl.dataset.name}-"
        f"faulty_clients[[]]-"
        f"TClients{cfg.tool.tracefl.data_dist.num_clients}-"
        f"{cfg.tool.tracefl.strategy.name}-(R{cfg.tool.tracefl.strategy.num_rounds}"
        f"-clientsPerR{cfg.tool.tracefl.strategy.clients_per_round})"
        f"-{cfg.tool.tracefl.data_dist.dist_type}"
        f"{cfg.tool.tracefl.data_dist.dirichlet_alpha}"
        f"-batch{cfg.tool.tracefl.data_dist.batch_size}"
        f"-epochs{cfg.tool.tracefl.client.epochs}-"
        f"lr{cfg.tool.tracefl.client.lr}"
    )
    return non_dp_key


def get_prov_eval_metrics(labels, predicted_labels):
    """Compute evaluation metrics for provenance by comparing true and predicted labels.

    Parameters
    ----------
    labels : array-like
        True labels.
    predicted_labels : array-like
        Predicted labels.

    Returns
    -------
    dict
        A dictionary containing evaluation metrics such as "Accuracy".
    """
    accuracy = metrics.accuracy_score(labels, predicted_labels)
    return {"Accuracy": accuracy}
