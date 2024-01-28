from functools import reduce
from typing import Any, Callable, List, Tuple

import numpy as np

from flwr.common import NDArray, NDArrays, parameters_to_ndarrays
from ..utils import flatten_params

def aggregate_dnc(results: List[Tuple[NDArrays, int]], c: float, niters: int, num_clients, num_malicious) -> NDArrays:
    flattened_params = [flatten_params(parameters_to_ndarrays(fitres.parameters)) for _, fitres in results]

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

        # save in I the self.sample_size[-1] - self.c * self.m[-1] smallestvalues from s
        # and return the indices of the smallest values
        to_keep = int(num_clients - c * num_malicious)
        I = np.argsort(np.array(s))[:to_keep]
        I_good.append(set(I.tolist()))

    I_final = list(set.intersection(*I_good))
    return np.array(results)[I_final].tolist()
