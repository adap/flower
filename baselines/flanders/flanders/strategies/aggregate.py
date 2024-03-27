"""Additional aggregation functions used as baselines against FLANDERS."""

from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import NDArrays
from flwr.server.strategy.aggregate import aggregate

from ..utils import flatten_params


def aggregate_dnc(
    results: List[Tuple[NDArrays, int]], c: float, niters: int, num_malicious: int
) -> NDArrays:
    """Aggregate fit results using DnC.

    Steps:
        1. Sample models
        2. Dimension-wise mean of parameters (mu)
        3. Compute the centered update (that is, model_i - mu -> model_c)
        4. SVD on model_c
        5. Compute outlier score
        6. Filter "bad" models.

    Parameters
    ----------
    results : List[Tuple[NDArrays, int]]
        The list of results from the clients.
    c : float
        The filtering fraction.
    niters : int
        The number of iterations.
    num_malicious : int
        The number of malicious clients.

    Returns
    -------
    NDArrays
        The aggregated parameters.
    """
    num_clients = len(results)
    flattened_params = [flatten_params(params) for params, _ in results]
    I_good = []
    for _ in range(niters):
        mu: NDArrays = [
            reduce(np.add, layer_updates) / len(flattened_params)
            for layer_updates in zip(*flattened_params)
        ]

        model_c = []
        for idx in range(len(flattened_params)):
            model_c.append(np.array(flattened_params[idx]) - np.array(mu))
        _, _, v = np.linalg.svd(model_c, full_matrices=False)
        s = [np.inner(model_i, v[0, :]) ** 2 for model_i in flattened_params]

        # save in I the num_clients - c * num_malicious smallestvalues from s
        # and return the indices of the smallest values
        to_keep = int(num_clients - c * num_malicious)
        I_set = np.argsort(np.array(s))[:to_keep]
        I_good.append(set(I_set.tolist()))

    I_final = list(set.intersection(*I_good))
    # Keep only the good models indicated by I_final in results
    res = [results[i] for i in I_final]
    aggregated_params = aggregate(res)
    return aggregated_params
