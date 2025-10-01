# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
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
"""Federated Learning Detector (FLDetector) [Zhang et al. 2023] Strategy.

Paper: https://dl.acm.org/doi/pdf/10.1145/3534678.3539231
"""
from typing import Union

from flwr.common import FitRes
from flwr.server.client_proxy import ClientProxy

import numpy as np
from sklearn.cluster import KMeans
from functools import reduce
import operator

from .client_results_strategy import ClientResultsStrategy


class FLDetector(ClientResultsStrategy):
    """FLDetector Strategy.

    Implementation based on https://dl.acm.org/doi/pdf/10.1145/3534678.3539231
    converted from MXNet into Numpy
    Original implementation: https://github.com/zaixizhang/FLDetector/tree/main
    """

    def __init__(self, number_of_workers: int):
        self.malicious_score = np.zeros((1, number_of_workers))
        self.history_length = 10
        self.grad_history = []
        self.weights_history = []

    def validate_client_results(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[
        list[tuple[ClientProxy, FitRes]],
        list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ]:
        """Scan client training results for malicious activity.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        trusted_results : Tuple[
            List[tuple[ClientProxy, FitRes]],
            List[Union[tuple[ClientProxy, FitRes], BaseException]
        ]
            The tuple represents the results that should be used for
            the next evaluation of the model training.
        """
        weights = [x.parameters.tensors for x in results]
        grads = np.mean(np.concatenate(*weights, axis=1), axis=-1, keepdims=1)

        self.weights_history.append(weights)
        self.grad_history.append(grads)

        self.weights_history = self.weights_history[-self.history_length-1:]
        self.grad_history = self.grad_history[-self.history_length-1:]

        if server_round > 50:
            if len(self.weights_history) > 2:
                raise ValueError("Expected to have history in epoch 50 and above")

            last_weight_diff = weights - self.weights_history[1]
            hvp = lbfgs(self.weights_history, self.grad_history, last_weight_diff)
        else:
            hvp = None
        distance = simple_mean(grads, weights, hvp)
        self.malicious_score = np.row_stack((self.malicious_score, distance))
        if self.malicious_score.shape[0] < self.history_length:
            return results, failures

        is_malicious = detection1(self.malicious_score)
        if is_malicious:
            return [], results + failures
        return results, failures


def lbfgs(S_k_list, Y_k_list, v):
    """L-BFGS, Limited Memory Broyden–Fletcher–Goldfarb–Shanno implementation."""
    curr_S_k = np.concatenate(*S_k_list, axis=1)
    curr_Y_k = np.concatenate(*Y_k_list, axis=1)
    S_k_time_Y_k = np.dot(curr_S_k.T, curr_Y_k)
    S_k_time_S_k = np.dot(curr_S_k.T, curr_S_k)
    R_k = np.triu(S_k_time_Y_k)
    L_k = S_k_time_Y_k - np.array(R_k)
    sigma_k = np.dot(Y_k_list[-1].T, S_k_list[-1]) / (
        np.dot(S_k_list[-1].T, S_k_list[-1])
    )
    D_k_diag = np.diag(S_k_time_Y_k)
    upper_mat = np.concatenate(*[sigma_k * S_k_time_S_k, L_k], axis=1)
    lower_mat = np.concatenate(*[L_k.T, -np.diag(D_k_diag)], axis=1)
    mat = np.concatenate(*[upper_mat, lower_mat], axis=0)
    mat_inv = np.linalg.inv(mat)

    approx_prod = sigma_k * v
    p_mat = np.concatenate(
        *[np.dot(curr_S_k.T, sigma_k * v), np.dot(curr_Y_k.T, v)], axis=0
    )
    approx_prod -= np.dot(
        np.dot(np.concatenate(*[sigma_k * curr_S_k, curr_Y_k], axis=1), mat_inv), p_mat
    )

    return approx_prod


def detection1(score):
    """Detect malicious activity."""
    nrefs = 10
    ks = range(1, 8)
    gaps = np.zeros(len(ks))
    gapDiff = np.zeros(len(ks) - 1)
    sdk = np.zeros(len(ks))
    min = np.min(score)
    max = np.max(score)
    score = (score - min) / (max - min)
    for i, k in enumerate(ks):
        estimator = KMeans(n_clusters=k)
        estimator.fit(score.reshape(-1, 1))
        label_pred = estimator.labels_
        center = estimator.cluster_centers_
        Wk = np.sum(
            [np.square(score[m] - center[label_pred[m]]) for m in range(len(score))]
        )
        WkRef = np.zeros(nrefs)
        for j in range(nrefs):
            rand = np.random.uniform(0, 1, len(score))
            estimator = KMeans(n_clusters=k)
            estimator.fit(rand.reshape(-1, 1))
            label_pred = estimator.labels_
            center = estimator.cluster_centers_
            WkRef[j] = np.sum(
                [np.square(rand[m] - center[label_pred[m]]) for m in range(len(rand))]
            )
        gaps[i] = np.log(np.mean(WkRef)) - np.log(Wk)
        sdk[i] = np.sqrt((1.0 + nrefs) / nrefs) * np.std(np.log(WkRef))

        if i > 0:
            gapDiff[i - 1] = gaps[i - 1] - gaps[i] + sdk[i]
    for i in range(len(gapDiff)):
        if gapDiff[i] >= 0:
            select_k = i + 1
            break

    if select_k == 1:
        print("No attack detected!")
        return False

    print("Attack Detected!")
    return True


def simple_mean(old_gradients, weights, hvp):
    """Rewrite of simple mean for tensors."""
    if hvp is not None:
        return None

    pred_grad = []
    distance = []
    for i in range(len(old_gradients)):
        pred_grad.append(old_gradients[i] + hvp)
    distance = np.norm(
        (np.concatenate(*pred_grad, axis=1) - np.concatenate(*weights, axis=1)), axis=0
    ).asnumpy()
    distance = np.sum(distance)
    return distance
