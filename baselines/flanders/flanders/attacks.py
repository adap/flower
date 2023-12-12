import math
import numpy as np
from typing import Dict, List, Tuple
from flwr.common import (
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    FitRes,
    NDArrays,
    NDArray,
    Parameters,
)
from flwr.server.client_proxy import ClientProxy
from scipy.stats import norm
#from strategy.krum import krum, _compute_distances

def no_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool], 
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    return ordered_results, {}

def gaussian_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool], 
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    magnitude = kwargs.get("magnitude", 0.0)
    dataset_name = kwargs.get("dataset_name", "no name")
    results = ordered_results.copy()
    perturbate = lambda a: a + np.random.normal(loc=0, scale=magnitude, size=len(a))
    
    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            params = parameters_to_ndarrays(fitres.parameters)
            if dataset_name == "income":
                new_params = [perturbate(layer) for layer in params]
            else:
                new_params = []
                for p in params:
                    new_params.append(np.apply_along_axis(perturbate, 0, p))
            fitres.parameters = ndarrays_to_parameters(new_params)
            results[int(fitres.metrics['cid'])] = (proxy, fitres)
    return results, {}

def lie_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool],
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    """
    Implementation of Omniscent LIE attack, Baruch et al. (2019)
    """
    results = ordered_results.copy()
    params = [parameters_to_ndarrays(fitres.parameters) for _, fitres in results]
    grads_mean = [np.mean(layer, axis=0) for layer in zip(*params)]
    grads_stdev = [np.std(layer, axis=0) ** 0.5 for layer in zip(*params)]

    n = len(ordered_results)                                        # number of clients
    m = sum(val == True for val in states.values())                 # number of corrupted clients
    s = math.floor((n / 2) + 1) - m                                 # number of supporters
    if s < 0:
        s = 1
    z_max = norm.ppf((n - m - s) / (n - m))

    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            mul_std = [layer * z_max for layer in grads_stdev]
            new_params = [grads_mean[i] - mul_std[i] for i in range(len(grads_mean))]
            fitres.parameters = ndarrays_to_parameters(new_params)
            results[int(fitres.metrics['cid'])] = (proxy, fitres)
    return results, {}

def fang_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool],
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    """
    Implemetation of Local Model Poisoning Attacks to Byzantine-Robust Federated Learning, Fang et al. (2020)
    Specifically designed for Krum, but they claim it works for other aggregation functions as well.
    Full-knowledge version (attackers knows the local models of all clients).

    Input:
        ordered_results - list of tuples (client_proxy, fit_result) ordered by client id
        states - dictionary of client ids and their states (True if malicious, False otherwise)
        magnitude - magnitude of the attack
        d - number of parameters
        w_re - selected model
        old_lambda - lambda value
        threshold - threshold for lambda
        malicious_selected - whether the malicious client was selected or not
    """

    d = kwargs.get("d", 1)
    w_re = kwargs.get("w_re", None)                                 # the received global model
    old_lambda = kwargs.get("old_lambda", 0.0)
    threshold = kwargs.get("threshold", 0.0)
    malicious_selected = kwargs.get("malicious_selected", False)
    agr_func = kwargs.get("agr_function", None)
    
    n = len(ordered_results)                                        # number of clients
    c = sum(val == True for val in states.values())                 # number of corrupted clients
    if c < 2:
        c = 2

    # lambda initialization
    if old_lambda == 0:
        benign = [
            (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
            for _, fitres in ordered_results if states[fitres.metrics["cid"]] == False
        ]
        all = [
            (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
            for _, fitres in ordered_results
        ]
        # Compute the smallest distance that Krum would choose
        _, _, _, distances = krum(all, c, 1)

        idx_benign = [int(cid) for cid in states.keys() if states[cid]==False]
        
        min_dist = np.min(np.array(distances)[idx_benign]) / ((n - 2*c - 1)*np.sqrt(d))

        # Compute max distance from w_re
        dist_wre = np.zeros((len(benign)))
        for i in range(len(benign)):
            dist = [benign[i][0][j] - w_re[j] for j in range(d)]
            norm_sums = 0
            for k in dist:
                norm_sums += np.linalg.norm(k)
            dist_wre[i] = norm_sums**2

        max_dist = np.max(dist_wre) / np.sqrt(d)
        l = min_dist + max_dist                                         # lambda
    else:
        # lambda halving search
        l = old_lambda
        if old_lambda > threshold and malicious_selected == False:
            l = old_lambda * 0.5

    # Compute sign vector s
    magnitude = []
    for i in range(len(w_re)):
        magnitude.append(np.sign(w_re[i]) * l)

    w_1 = [w_re[i] - magnitude[i] for i in range(len(w_re))]            # corrupted model
    corrupted_params = ndarrays_to_parameters(w_1)

    # Set corrupted clients' updates to w_1
    results =[
        (
            proxy, 
            FitRes(fitres.status, parameters=corrupted_params, num_examples=fitres.num_examples, metrics=fitres.metrics)
        ) if states[fitres.metrics["cid"]] else (proxy, fitres) 
        for proxy, fitres in ordered_results
    ]

    return results, {"lambda": l}

def minmax_attack(
        ordered_results:List[Tuple[ClientProxy, FitRes]], 
        states:Dict[str, bool], 
        **kwargs
    ) -> List[Tuple[ClientProxy, FitRes]]:
    """
    Implementation of Min-Max agnostic (O == True according to the heuristic) 
    attack (full-knowledge, perturbation function chosen according to our 
    experimental results).
    
    From: 
    "Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning"
    (Shejwalkar et al., 2021)
    """
    dataset_name = kwargs.get("dataset_name", None)
    threshold = kwargs.get("threshold", 1e-5)
    lambda_init = kwargs.get("old_lambda", 5.0)
    malicious_num = kwargs.get("malicious_num", 0)

    results = ordered_results.copy()
    params = [parameters_to_ndarrays(fitres.parameters) for _, fitres in results]
    params_avg = [np.mean(param, axis=0) for param in zip(*params)]

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
    l = lambda_init
    step = lambda_init * 0.5
    while abs(lambda_succ - l) > threshold and step > threshold and malicious_num > 0:
        # Compute malicious gradients
        pv = [l * perturbation_vect[i] for i in range(len(perturbation_vect))]
        corrupted_params = [params_avg[i] + pv[i] for i in range(len(params_avg))]

        # Set corrupted clients' updates to corrupted_params
        params_c = [corrupted_params if states[i] else params[i] for i in range(len(params))]
        M = _compute_distances(params_c)

        # Remove from matrix M all malicious clients in both rows and columns
        M_b = np.delete(M, [i for i in range(len(M)) if states[results[i][1].metrics["cid"]]], axis=0)
        M_b = np.delete(M_b, [i for i in range(len(M)) if states[results[i][1].metrics["cid"]]], axis=1)

        # Remove from M all benign clients on rows and all malicious on columns
        M_m = np.delete(M, [i for i in range(len(M)) if not states[results[i][1].metrics["cid"]]], axis=0)
        M_m = np.delete(M_m, [i for i in range(len(M)) if states[results[i][1].metrics["cid"]]], axis=1)

        # Take the maximum distance between any benign client and any malicious one
        max_dist_m = np.max(M_m)

        # Take the maximum distance between any two benign clients
        max_dist_b = np.max(M_b)

        # Compute lambda (best scaling coefficient)
        if max_dist_m < max_dist_b:
            # Lambda (gamma in the paper) is good. Save and try to increase it
            lambda_succ = l
            l = l + step * 0.5
        else:
            # Lambda is to big, must be reduced to increse the chances of being selected
            l = l - step * 0.5
        step *= 0.5

    # Compute the final malicious update
    perturbation_vect = [lambda_succ * perturbation_vect[i] for i in range(len(perturbation_vect))]
    corrupted_params = [params_avg[i] + perturbation_vect[i] for i in range(len(params_avg))]
    corrupted_params = ndarrays_to_parameters(corrupted_params)
    for proxy, fitres in ordered_results:
        if states[fitres.metrics["cid"]]:
            fitres.parameters = corrupted_params
            results[int(fitres.metrics['cid'])] = (proxy, fitres)
    return results, {"lambda": l}

def krum(results: List[Tuple[List, int]], m: int, to_keep: int, num_closest=None):
    """
    Get the best parameters vector according to the Krum function.
    Output: the best parameters vector.
    """
    weights = [w for w, _ in results]                                       # list of weights
    M = _compute_distances(weights)                                         # matrix of distances

    if not num_closest:
        num_closest = len(weights) - m - 2                                  # number of closest points to use
    if num_closest <= 0:
        num_closest = 1
    elif num_closest > len(weights):
        num_closest = len(weights)

    closest_indices = _get_closest_indices(M, num_closest)                  # indices of closest points
    scores = [np.sum(M[i,closest_indices[i]]) for i in range(len(M))]       # scores i->j for each i
    
    best_index = np.argmin(scores)                                          # index of the best score
    best_indices = np.argsort(scores)[::-1][len(scores)-to_keep:]           # indices of best scores (multikrum)
    return weights[best_index], best_index, best_indices, scores


def _compute_distances(weights: List[NDArrays]) -> NDArray:
    """Compute distances between vectors.

    Input: weights - list of weights vectors
    Output: distances - matrix distance_matrix of squared distances between the vectors
    """
    flat_w = np.array([np.concatenate(p, axis=None).ravel() for p in weights])
    distance_matrix = np.zeros((len(weights), len(weights)))
    for i, _ in enumerate(flat_w):
        for j, _ in enumerate(flat_w):
            delta = flat_w[i] - flat_w[j]
            norm = np.linalg.norm(delta)
            distance_matrix[i, j] = norm**2
    return distance_matrix

def _get_closest_indices(M, num_closest: int) -> List[int]:
    """
    Get the indices of the closest points.
    Input: 
        M - matrix of squared distances between the vectors
        num_closest - number of closest points to get for each parameter vector
    Output:
        closest_indices - list of lists of indices of the closest points for each parameter vector 
    """
    closest_indices = []
    for i in range(len(M)):
        closest_indices.append(np.argsort(M[i])[1:num_closest+1].tolist())
    return closest_indices