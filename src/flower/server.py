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
"""Flower server"""


import concurrent.futures
import random
from functools import reduce
from typing import List, Tuple

import numpy as np

from flower.client import Client
from flower.typing import Weights


class Server:
    """Flower server"""

    def __init__(self, clients: List[Client]):
        self.weights: Weights = []
        self.clients: List[Client] = clients

    def fit(self, num_rounds: int) -> None:
        """Run federated averaging for a number of rounds"""
        # Initialize weights by asking one client to return theirs
        self.weights = self._get_initial_weights()

        # Run federated averaging for num_rounds
        for current_round in range(num_rounds):
            # Refine model
            self.fit_round()
            # Evaluate refined model
            loss_avg = self.evaluate()
            print(f"Round {current_round}: weighted loss average {loss_avg}")

    def evaluate(self) -> float:
        """Validate current global model on a number of clients"""
        # Sample clients for evaluation
        clients = random.sample(self.clients, 3)

        # Evaluate current global weights on those clients
        results = eval_clients(clients, self.weights)

        # Aggregate the evaluation results
        return weighted_loss_avg(results)

    def fit_round(self) -> None:
        """Perform a single round of federated averaging"""
        # Sample three clients
        clients = random.sample(self.clients, 3)

        # Collect training results from all clients participating in this round
        results = fit_clients(clients, self.weights)

        # Aggregate training results and replace previous global model
        weights_prime = aggregate(results)
        self.weights = weights_prime

    def _get_initial_weights(self) -> Weights:
        """Get initial weights from one of the available clients"""
        return random.choice(self.clients).get_weights()


def fit_clients(clients: List[Client], weights: Weights) -> List[Tuple[Weights, int]]:
    """Refine weights concurrently on all selected clients"""
    results: List[Tuple[Weights, int]] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fit_client, c, weights) for c in clients]
        concurrent.futures.wait(futures)
        for future in futures:
            results.append(future.result())
    return results


def fit_client(client: Client, weights: Weights) -> Tuple[Weights, int]:
    """Refine weights on a single client"""
    return client.fit(weights)


def aggregate(results: List[Tuple[Weights, int]]) -> Weights:
    """Compute weighted average"""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def eval_clients(clients: List[Client], weights: Weights) -> List[Tuple[int, float]]:
    """Evaluate weights concurrently on all selected clients"""
    results: List[Tuple[int, float]] = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(eval_client, c, weights) for c in clients]
        concurrent.futures.wait(futures)
        for future in futures:
            results.append(future.result())
    return results


def eval_client(client: Client, weights: Weights) -> Tuple[int, float]:
    """Evaluate weights on a single client"""
    return client.evaluate(weights)


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients"""
    print(f"results: {results}")
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples
