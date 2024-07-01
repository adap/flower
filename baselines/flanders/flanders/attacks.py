"""Implementation of attacks used in the paper."""

import math
from typing import Dict, List, Tuple

import numpy as np
from flwr.common import FitRes, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from scipy.stats import norm


# pylint: disable=unused-argument
def no_attack(
    ordered_results: List[Tuple[ClientProxy, FitRes]], states: Dict[str, bool], **kwargs
):
    """No attack."""
    return ordered_results, {}


def gaussian_attack(ordered_results, states, **kwargs):
    """Apply Gaussian attack on parameters.

    Parameters
    ----------
    ordered_results
        List of tuples (client_proxy, fit_result) ordered by client id.
    states
        Dictionary of client ids and their states (True if malicious, False otherwise).
    magnitude
        Magnitude of the attack.
    dataset_name
        Name of the dataset.

    Returns
    -------
    results
        List of tuples (client_proxy, fit_result) ordered by client id.
    """
    magnitude = kwargs.get("magnitude", 0.0)
    dataset_name = kwargs.get("dataset_name", "no name")
    results = ordered_results.copy()

    def perturbate(vect):
        return vect + np.random.normal(loc=0, scale=magnitude, size=vect.size)

    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            params = parameters_to_ndarrays(fitres.parameters)
            if dataset_name == "income":
                new_params = [perturbate(layer) for layer in params]
            else:
                new_params = []
                for par in params:
                    # if par is an array of one element, it is a scalar
                    if par.size == 1:
                        new_params.append(perturbate(par))
                    else:
                        new_params.append(np.apply_along_axis(perturbate, 0, par))
            fitres.parameters = ndarrays_to_parameters(new_params)
            results[int(fitres.metrics["cid"])] = (proxy, fitres)
    return results, {}


# pylint: disable=too-many-locals, unused-argument
def lie_attack(
    ordered_results,
    states,
    omniscent=True,
    **kwargs,
):
    """Apply Omniscent LIE attack, Baruch et al. (2019) on parameters.

    Parameters
    ----------
    ordered_results
        List of tuples (client_proxy, fit_result) ordered by client id.
    states
        Dictionary of client ids and their states (True if malicious, False otherwise).
    omniscent
        Whether the attacker knows the local models of all clients or not.

    Returns
    -------
    results
        List of tuples (client_proxy, fit_result) ordered by client id.
    """
    results = ordered_results.copy()
    params = [parameters_to_ndarrays(fitres.parameters) for _, fitres in results]
    grads_mean = [np.mean(layer, axis=0) for layer in zip(*params)]
    grads_stdev = [np.std(layer, axis=0) ** 0.5 for layer in zip(*params)]

    if not omniscent:
        # if not omniscent, the attacker doesn't know the
        # local models of all clients, but only of the corrupted ones
        params = [
            params[i]
            for i in range(len(params))
            if states[results[i][1].metrics["cid"]]
        ]

    num_clients = len(ordered_results)
    num_malicious = sum(val is True for val in states.values())

    # pylint: disable=c-extension-no-member
    num_supporters = math.floor((num_clients / 2) + 1) - num_malicious

    z_max = norm.cdf(
        (num_clients - num_malicious - num_supporters) / (num_clients - num_malicious)
    )

    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            mul_std = [layer * z_max for layer in grads_stdev]
            new_params = [grads_mean[i] - mul_std[i] for i in range(len(grads_mean))]
            fitres.parameters = ndarrays_to_parameters(new_params)
            results[int(fitres.metrics["cid"])] = (proxy, fitres)
    return results, {}


def fang_attack(
    ordered_results,
    states,
    omniscent=True,
    **kwargs,
):
    """Apply Local Model Poisoning Attacks.

    (Fang et al. (2020))
    Specifically designed for Krum, but they claim it works for other
    aggregation functions as well.
    Full-knowledge version (attackers knows the local models of all clients).

    Parameters
    ----------
    ordered_results
        List of tuples (client_proxy, fit_result) ordered by client id.
    states
        Dictionary of client ids and their states (True if malicious, False
        otherwise).
    omniscent
        Whether the attacker knows the local models of all clients or not.
    num_layers
        Number of layers.
    w_re
        The received global model.
    old_lambda
        The lambda from the previous round.
    threshold
        The threshold for lambda.
    malicious_selected
        Whether the attacker was selected as malicious in the previous round.

    Returns
    -------
    results
        List of tuples (client_proxy, fit_result) ordered by client id.
    """
    num_layers = kwargs.get("num_layers", 2)
    w_re = kwargs.get("w_re", None)  # the received global model
    threshold = kwargs.get("threshold", 1e-5)

    num_clients = len(ordered_results)
    num_corrupted = sum(val is True for val in states.values())
    # there can't be an attack with less than 2 malicious clients
    # to avoid division by 0
    num_corrupted = max(num_corrupted, 2)

    if not omniscent:
        # if not omniscent, the attacker doesn't know the
        # local models of all clients, but only of the corrupted ones
        ordered_results = [
            ordered_results[i]
            for i in range(len(ordered_results))
            if states[ordered_results[i][1].metrics["cid"]]
        ]

    # Initialize lambda
    benign = [
        (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
        for _, fitres in ordered_results
        if states[fitres.metrics["cid"]] is False
    ]
    all_params = [
        (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
        for _, fitres in ordered_results
    ]
    # Compute the smallest distance that Krum would choose
    _, _, _, distances = _krum(all_params, num_corrupted, 1)

    idx_benign = [int(cid) for cid in states.keys() if states[cid] is False]

    min_dist = np.min(np.array(distances)[idx_benign]) / (
        ((num_clients - 2) * (num_corrupted - 1)) * np.sqrt(num_layers)
    )

    # Compute max distance from w_re
    dist_wre = np.zeros((len(benign)))
    for i in range(len(benign)):
        dist = [benign[i][0][j] - w_re[j] for j in range(num_layers)]
        norm_sums = 0
        for k in dist:
            norm_sums += np.linalg.norm(k)
        dist_wre[i] = norm_sums**2
    max_dist = np.max(dist_wre) / np.sqrt(num_layers)
    lamda = min(
        min_dist + max_dist, 999
    )  # lambda (capped to 999 to avoid numerical problems in specific settings)

    malicious_selected, corrupted_params = _fang_corrupt_and_select(
        all_params, w_re, states, num_corrupted, lamda
    )
    while lamda > threshold and malicious_selected is False:
        lamda = lamda * 0.5
        malicious_selected, corrupted_params = _fang_corrupt_and_select(
            all_params, w_re, states, num_corrupted, lamda
        )

    # Set corrupted clients' updates to w_1
    results = [
        (
            (
                proxy,
                FitRes(
                    fitres.status,
                    parameters=ndarrays_to_parameters(corrupted_params),
                    num_examples=fitres.num_examples,
                    metrics=fitres.metrics,
                ),
            )
            if states[fitres.metrics["cid"]]
            else (proxy, fitres)
        )
        for proxy, fitres in ordered_results
    ]

    return results, {}


def minmax_attack(
    ordered_results,
    states,
    omniscent=True,
    **kwargs,
):
    """Apply Min-Max agnostic attack.

    Full-knowledge, perturbation function chosen according to our experimental
    results.
    From:
    "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and
    Defenses for Federated Learning" (Shejwalkar et al., 2021)

    Parameters
    ----------
    ordered_results
        List of tuples (client_proxy, fit_result) ordered by client id.
    states
        Dictionary of client ids and their states (True if malicious, False
        otherwise).
    omniscent
        Whether the attacker knows the local models of all clients or not.
    threshold
        Threshold for lambda.
    lambda_init
        Initial value for lambda.

    Returns
    -------
    results
        List of tuples (client_proxy, fit_result) ordered by client id.
    """
    dataset_name = kwargs.get("dataset_name", None)
    threshold = kwargs.get("threshold", 1e-5)
    lambda_init = kwargs.get("lambda", 5.0)
    malicious_num = kwargs.get("malicious_num", 0)

    results = ordered_results.copy()
    params = [parameters_to_ndarrays(fitres.parameters) for _, fitres in results]
    params_avg = [np.mean(param, axis=0) for param in zip(*params)]

    if not omniscent:
        # if not omniscent, the attacker doesn't know the
        # local models of all clients, but only of the corrupted ones
        results = [
            results[i]
            for i in range(len(results))
            if states[results[i][1].metrics["cid"]]
        ]

    # Decide what perturbation to use according to the
    # results presented in the paper.
    if dataset_name == "mnist":
        # Apply std perturbation
        # In the paper authors state that sign function is the best
        # but in my experience std perturbation works better
        perturbation_vect = [-np.std(layer, axis=0) for layer in zip(*params)]
    elif dataset_name == "cifar":
        # Apply std perturbation
        perturbation_vect = [-np.std(layer, axis=0) for layer in zip(*params)]
    else:
        # Apply std perturbation
        perturbation_vect = [-np.std(layer, axis=0) for layer in zip(*params)]

    # Compute lambda (referred as gamma in the paper)
    lambda_succ = lambda_init + 1
    curr_lambda = lambda_init
    step = lambda_init * 0.5
    while (
        abs(lambda_succ - curr_lambda) > threshold
        and step > threshold
        and malicious_num > 0
    ):
        # Compute malicious gradients
        perturbed_params = [
            curr_lambda * perturbation_vect[i] for i in range(len(perturbation_vect))
        ]
        corrupted_params = [
            params_avg[i] + perturbed_params[i] for i in range(len(params_avg))
        ]

        # Set corrupted clients' updates to corrupted_params
        params_c = [
            corrupted_params if states[str(i)] else params[i]
            for i in range(len(params))
        ]
        distance_matrix = _compute_distances(params_c)

        # Remove from matrix distance_matrix all malicious clients in both
        # rows and columns
        distance_matrix_b = np.delete(
            distance_matrix,
            [
                i
                for i in range(len(distance_matrix))
                if states[results[i][1].metrics["cid"]]
            ],
            axis=0,
        )
        distance_matrix_b = np.delete(
            distance_matrix_b,
            [
                i
                for i in range(len(distance_matrix))
                if states[results[i][1].metrics["cid"]]
            ],
            axis=1,
        )

        # Remove from distance_matrix all benign clients on
        # rows and all malicious on columns
        distance_matrix_m = np.delete(
            distance_matrix,
            [
                i
                for i in range(len(distance_matrix))
                if not states[results[i][1].metrics["cid"]]
            ],
            axis=0,
        )
        distance_matrix_m = np.delete(
            distance_matrix_m,
            [
                i
                for i in range(len(distance_matrix))
                if states[results[i][1].metrics["cid"]]
            ],
            axis=1,
        )

        # Take the maximum distance between any benign client and any malicious one
        max_dist_m = np.max(distance_matrix_m)

        # Take the maximum distance between any two benign clients
        max_dist_b = np.max(distance_matrix_b)

        # Compute lambda (best scaling coefficient)
        if max_dist_m < max_dist_b:
            # Lambda (gamma in the paper) is good. Save and try to increase it
            lambda_succ = curr_lambda
            curr_lambda = curr_lambda + step * 0.5
        else:
            # Lambda is to big, must be reduced to increse the chances of being selected
            curr_lambda = curr_lambda - step * 0.5
        step *= 0.5

    # Compute the final malicious update
    perturbation_vect = [
        lambda_succ * perturbation_vect[i] for i in range(len(perturbation_vect))
    ]
    corrupted_params = [
        params_avg[i] + perturbation_vect[i] for i in range(len(params_avg))
    ]
    corrupted_params = ndarrays_to_parameters(corrupted_params)
    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            fitres.parameters = corrupted_params
            results[int(fitres.metrics["cid"])] = (proxy, fitres)
    return results, {}


def _krum(results, num_malicious, to_keep, num_closest=None):
    """Get the best parameters vector according to the Krum function.

    Output: the best parameters vector.
    """
    weights = [w for w, _ in results]  # list of weights
    distance_matrix = _compute_distances(weights)  # matrix of distances

    if not num_closest:
        num_closest = (
            len(weights) - num_malicious - 2
        )  # number of closest points to use
    if num_closest <= 0:
        num_closest = 1
    elif num_closest > len(weights):
        num_closest = len(weights)

    closest_indices = _get_closest_indices(
        distance_matrix, num_closest
    )  # indices of closest points

    scores = [
        np.sum(distance_matrix[i, closest_indices[i]])
        for i in range(len(distance_matrix))
    ]  # scores i->j for each i

    best_index = np.argmin(scores)  # index of the best score
    best_indices = np.argsort(scores)[::-1][
        len(scores) - to_keep :
    ]  # indices of best scores (multikrum)
    return weights[best_index], best_index, best_indices, scores


def _compute_distances(weights):
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = np.array([np.concatenate(par, axis=None).ravel() for par in weights])
    distance_matrix = np.zeros((len(weights), len(weights)))
    for i, _ in enumerate(flat_w):
        for j, _ in enumerate(flat_w):
            delta = flat_w[i] - flat_w[j]
            dist = np.linalg.norm(delta)
            distance_matrix[i, j] = dist**2
    return distance_matrix


def _get_closest_indices(distance_matrix, num_closest):
    """Get the indices of the closest points.

    Args:
        distance_matrix
            matrix of distances
        num_closest
            number of closest points to get for each parameter vector
    Output:
        closest_indices
            list of lists of indices of the closest points for each vector.
    """
    closest_indices = []
    for idx, _ in enumerate(distance_matrix):
        closest_indices.append(
            np.argsort(distance_matrix[idx])[1 : num_closest + 1].tolist()
        )
    return closest_indices


def _fang_corrupt_params(global_model, lamda):
    # Compute sign vector num_supporters
    magnitude = []
    for i, _ in enumerate(global_model):
        magnitude.append(np.sign(global_model[i]) * lamda)

    corrupted_params = [
        global_model[i] - magnitude[i] for i in range(len(global_model))
    ]  # corrupted model
    return corrupted_params


def _fang_corrupt_and_select(all_models, global_model, states, num_corrupted, lamda):
    # Check that krum selects a malicious client
    corrupted_params = _fang_corrupt_params(global_model, lamda)
    all_models_m = [
        (corrupted_params, num_examples) if states[str(i)] else (model, num_examples)
        for i, (model, num_examples) in enumerate(all_models)
    ]
    _, idx_best_model, _, _ = _krum(all_models_m, num_corrupted, 1)

    # Check if the best model is malicious
    malicious_selected = states[str(idx_best_model)]
    return malicious_selected, corrupted_params
