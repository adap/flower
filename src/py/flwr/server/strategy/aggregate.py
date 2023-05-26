# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import Dict, List, Tuple

import numpy as np

from flwr.common import NDArray, NDArrays


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def aggregate_median(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute median."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute median weight of each layer
    median_w: NDArrays = [
        np.median(np.asarray(layer), axis=0) for layer in zip(*weights)  # type: ignore
    ]
    return median_w


def aggregate_krum(
    results: List[Tuple[NDArrays, int]], num_malicious: int, to_keep: int
) -> NDArrays:
    """Choose one parameter vector according to the Krum function.

    If to_keep is not None, then MultiKrum is applied.
    """
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    # Compute distances between vectors
    distance_matrix = _compute_distances(weights)

    # For each client, take the n-f-2 closest parameters vectors
    num_closest = max(1, len(weights) - num_malicious - 2)
    closest_indices = []
    for i, _ in enumerate(distance_matrix):
        closest_indices.append(
            np.argsort(distance_matrix[i])[1 : num_closest + 1].tolist()  # noqa: E203
        )

    # Compute the score for each client, that is the sum of the distances
    # of the n-f-2 closest parameters vectors
    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]

    if to_keep > 0:
        # Choose to_keep clients and return their average (MultiKrum)
        best_indices = np.argsort(scores)[::-1][len(scores) - to_keep :]  # noqa: E203
        best_results = [results[i] for i in best_indices]
        return aggregate(best_results)

    # Return the model parameters that minimize the score (Krum)
    return weights[np.argmin(scores)]


# pylint: disable=too-many-locals
def aggregate_bulyan(
    results: List[Tuple[NDArrays, int]], num_malicious: int
) -> NDArrays:
    """Perform Bulyan aggregation."""
    selected_models_set: Dict[int, Tuple[NDArrays, int]] = {}

    # List of idx to keep track of the order of clients
    tracker: NDArray = np.arange(len(results))

    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    theta = len(weights) - 2 * num_malicious
    if theta <= 0:
        theta = 1

    beta = theta - 2 * num_malicious
    if beta <= 0:
        beta = 1

    for _ in range(theta):
        best_model = aggregate_krum(results, num_malicious, to_keep=0)
        list_of_weights = [weights for weights, num_samples in results]
        best_idx = _find_reference_weights(best_model, list_of_weights)

        selected_models_set[tracker[best_idx]] = results[best_idx]

        # remove idx from tracker and weights_results
        tracker = np.delete(tracker, best_idx)  # type: ignore
        results.pop(best_idx)

    # Compute median parameter vector across selected_models_set
    median_vect = aggregate_median(list(selected_models_set.values()))

    # Take the beta closest params to the median
    distances = {}
    for idx, result in selected_models_set.items():
        dist = [
            np.abs(result[0][0][j] - median_vect[0][j])
            for j in range(len(weights[0][0]))
        ]
        norm_sums = 0
        for k in dist:
            norm_sums += np.linalg.norm(k)  # type: ignore
        distances[idx] = norm_sums

    closest_idx = sorted(distances, key=lambda idx: distances[idx])[:beta]
    closest_models_to_median = [selected_models_set[i][0] for i in closest_idx]

    # Apply FevAvg on closest_models_to_median
    parameters_aggregated: NDArrays = [
        reduce(np.add, layers) / beta for layers in zip(*closest_models_to_median)
    ]

    return parameters_aggregated


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    parameters: NDArrays, deltas: List[NDArrays], hs_fll: List[NDArrays]
) -> NDArrays:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_parameters = [(u - v) * 1.0 for u, v in zip(parameters, updates)]
    return new_parameters


def _compute_distances(weights: List[NDArrays]) -> NDArray:
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = np.array(
        [np.concatenate(p, axis=None).ravel() for p in weights]  # type: ignore
    )
    distance_matrix = np.zeros((len(weights), len(weights)))
    for i, _ in enumerate(flat_w):
        for j, _ in enumerate(flat_w):
            delta = flat_w[i] - flat_w[j]
            norm = np.linalg.norm(delta)  # type: ignore
            distance_matrix[i, j] = norm**2
    return distance_matrix


def _trim_mean(array: NDArray, proportiontocut: float) -> NDArray:
    """Compute trimmed mean along axis=0.

     It is based on the scipy implementation.

    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.stats.trim_mean.html.
    """
    axis = 0
    nobs = array.shape[axis]
    lowercut = int(proportiontocut * nobs)
    uppercut = nobs - lowercut
    if lowercut > uppercut:
        raise ValueError("Proportion too big.")

    atmp = np.partition(array, (lowercut, uppercut - 1), axis)

    slice_list = [slice(None)] * atmp.ndim
    slice_list[axis] = slice(lowercut, uppercut)
    result: NDArray = np.mean(atmp[tuple(slice_list)], axis=axis)
    return result


def aggregate_trimmed_avg(
    results: List[Tuple[NDArrays, int]], proportiontocut: float
) -> NDArrays:
    """Compute trimmed average."""
    # Create a list of weights and ignore the number of examples
    weights = [weights for weights, _ in results]

    trimmed_w: NDArrays = [
        _trim_mean(np.asarray(layer), proportiontocut=proportiontocut)
        for layer in zip(*weights)
    ]

    return trimmed_w


def _check_weights_equality(weights1: NDArrays, weights2: NDArrays) -> bool:
    """Check if weights are the same."""
    return all(
        np.array_equal(layer_weights1, layer_weights2)
        for layer_weights1, layer_weights2 in zip(weights1, weights2)
    )


def _find_reference_weights(
    reference_weights: NDArrays, list_of_weights: List[NDArrays]
) -> int:
    """Loop through the `list_of_weights` to find the reference weights, raise
    Error if not found.

    Parameters
    ----------
    reference_weights: NDArrays
        Weights that will be searched for.
    list_of_weights: List[NDArrays]
        List of weights that will be searched through.

    Returns
    -------
    index: int
        The index of `reference_weights` in the `list_of_weights`.

    Raises
    ------
    ValueError
        If `reference_weights` is not found in `list_of_weights`.
    """
    for idx, weights in enumerate(list_of_weights):
        if _check_weights_equality(reference_weights, weights):
            return idx
    raise ValueError("The reference weights not found in list_of_weights.")
