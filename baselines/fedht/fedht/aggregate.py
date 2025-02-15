"""Aggregation functions for strategy implementations."""

# mypy: disallow_untyped_calls=False

from functools import reduce
from typing import List, Tuple

import numpy as np
from flwr.common import NDArrays

# calls hardthreshold function for each list element in weights_all
def hardthreshold_list(weights_all, num_keep: int) -> NDArrays:
    """Call hardthreshold."""

    params = [
        hardthreshold(each, num_keep) for each in weights_all
    ]
    return [params]

# hardthreshold function applied to array
def hardthreshold(weights_prime, num_keep: int) -> NDArrays:
    """Perform hardthresholding on single array."""
    # check for len of array
    val_len = weights_prime.size

    # intercepts not hardthresholded
    if val_len > 1:
        if num_keep > val_len:
            params = weights_prime
            print(
                "num_keep parameter greater than length of vector. All parameters kept."
            )
        else:
            # Compute the magnitudes
            magnitudes = np.abs(weights_prime)

            # Get the k-th largest value in the vector
            threshold = np.partition(magnitudes, -num_keep)[-num_keep]

            # Create a new vector where values below the threshold are set to zero
            params = np.where(magnitudes >= threshold, weights_prime, 0)

    else:
        params = weights_prime

    return np.array(params)

# hardthreshold aggregation
def aggregate_hardthreshold(
    results: List[Tuple[NDArrays, int]], num_keep: int, iterht: bool
) -> NDArrays:
    """Apply hard thresholding to keep only the k largest weights.

    Fed-HT (Fed-IterHT) can be found at
    https://arxiv.org/abs/2101.00052
    """
    if num_keep <= 0:
        raise ValueError("k must be a positive integer.")

    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum(num_examples for (_, num_examples) in results)

    green = "\033[92m"
    reset = "\033[0m"

    # check for iterht=True; set in cfg
    if iterht:
        print(
            f"{green}INFO {reset}:\t\tUsing Fed-IterHT with num_keep = ",
            num_keep,
        )

        # apply across all models within each client

        # 'results' is a collection of tuples of the form (params, num_obs)
        # we want to adjust the 'params' portion of that tuple
        # i: iterates through clients
        # ignoring second element in results[i] skips over number of observations
        # j: iterates through all layers of a model
        # k: iterates through the slices of each layer
        for i in range(len(results)):
            for j in range(len(results[i][0])):
                for k in range(len(results[i][0][j])):
                    results[i][0][j][k] = hardthreshold(results[i][0][j][k], num_keep)

        weighted_weights1 = [
            [layer * num_examples for layer in weights]
            for weights, num_examples in results
        ]
        weighted_weights2 = weighted_weights1

    else:
        print(
            f"{green}INFO {reset}:\t\tUsing Fed-HT with num_keep = ",
            num_keep,
        )

        weighted_weights1 = [
            [layer * num_examples for layer in weights]
            for weights, num_examples in results
        ]
        weighted_weights2 = weighted_weights1

    hold = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights2)
    ]

    new_result: NDArrays = [
        val
        for sublist in (
            hardthreshold_list(layer_updates, num_keep) for layer_updates in hold
        )
        for val in sublist
    ]

    return new_result
