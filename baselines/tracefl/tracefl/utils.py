"""Define any utility function.

They are not directly relevant to  the other (more FL specific) python
modules. For example, you may define here things like: loading a model
from a checkpoint, saving results, plotting.
"""

import numpy as np
from sklearn import metrics


def get_backend_config(cfg):
    client_resources = {
        "num_cpus": cfg.tool.tracefl.client_resources.cpus,
        "num_gpus": cfg.tool.tracefl.client_resources.gpus,
    }
    return {
        "client_resources": client_resources,
        "init_args": {"log_to_driver": True, "logging_level": 30},
        "actor": {"tensorflow": 0},
    }


def compute_importance(n_elements, decay_factor=0.9):
    """Compute importance weights for a sequence of elements using a decay
    factor.

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
    """Generate a unique experiment key based on the configuration settings.

    Parameters
    ----------
    cfg : object
        Configuration object containing experiment parameters such as noise_multiplier, clipping_norm,
        model name, dataset name, faulty_clients_ids, noise_rate, number of clients, strategy, data distribution,
        batch size, epochs, and learning rate.

    Returns
    -------
    str
        A unique experiment key string.

    Raises
    ------
    ValueError
        If the configuration is invalid.
    """
    if (
        cfg.tool.tracefl.strategy.noise_multiplier == -1
        and cfg.tool.tracefl.strategy.clipping_norm == -1
    ):
        key = (
            f"{cfg.tool.tracefl.exp_key}-"
            f"{cfg.tool.tracefl.model.name}-{cfg.tool.tracefl.dataset.name}-"
            f"noise_rate{cfg.tool.tracefl.strategy.noise_rate}-"
            f"TClients{cfg.tool.tracefl.data_dist.num_clients}-"
            f"{cfg.tool.tracefl.strategy.name}-(R{cfg.tool.tracefl.strategy.num_rounds}"
            f"-clientsPerR{cfg.tool.tracefl.strategy.clients_per_round})"
            f"-{cfg.tool.tracefl.data_dist.dist_type}{cfg.tool.tracefl.data_dist.dirichlet_alpha}"
            f"-batch{cfg.tool.tracefl.data_dist.batch_size}-epochs{cfg.tool.tracefl.client.epochs}-"
            f"lr{cfg.tool.tracefl.client.lr}"
        )
        print("Line 80")
        return key

    else:
        raise ValueError("Invalid config")


def get_prov_eval_metrics(labels, predicted_labels):
    """Compute evaluation metrics for provenance by comparing true and
    predicted labels.

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
    answer = {
        "Accuracy": accuracy,
    }

    return answer
