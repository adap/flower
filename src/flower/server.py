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
"""Flower server."""


import concurrent.futures
from functools import reduce
from typing import List, Optional, Tuple

import numpy as np

from flower.client import Client
from flower.client_manager import ClientManager
from flower.history import History
from flower.strategy import DefaultStrategy, Strategy
from flower.typing import Weights


class Server:
    """Flower server."""

    def __init__(
        self, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.weights: Weights = []
        self.strategy: Strategy = strategy if strategy is not None else DefaultStrategy()

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    def fit(self, num_rounds: int) -> History:
        """Run federated averaging for a number of rounds"""
        # Initialize weights by asking one client to return theirs
        self.weights = self._get_initial_weights()
        # Run federated averaging for num_rounds
        history = History()
        for current_round in range(num_rounds):
            # Refine model
            self.fit_round()
            # Evaluate refined model
            if self.strategy.should_evaluate():
                loss_avg = self.evaluate()
                history.add_loss(current_round, loss_avg)
            # Inform strategy that we're moving on to the next round
            self.strategy.next_round()
        return history

    def evaluate(self) -> float:
        """Validate current global model on a number of clients"""
        # Sample clients for evaluation
        sample_size = self.strategy.num_evaluation_clients(
            self._client_manager.num_available()
        )
        clients = self._client_manager.sample(sample_size)

        # Evaluate current global weights on those clients
        results = eval_clients(clients, self.weights)

        # Aggregate the evaluation results
        return weighted_loss_avg(results)

    def fit_round(self) -> None:
        """Perform a single round of federated averaging"""
        # Sample a number of clients (dependent on the strategy)
        sample_size = self.strategy.num_evaluation_clients(
            self._client_manager.num_available()
        )
        clients = self._client_manager.sample(sample_size)

        # Collect training results from all clients participating in this round
        results = fit_clients(clients, self.weights)

        # Aggregate training results and replace previous global model
        weights_prime = aggregate(results)
        self.weights = weights_prime

    def _get_initial_weights(self) -> Weights:
        """Get initial weights from one of the available clients"""
        random_client = self._client_manager.sample(1)[0]
        return random_client.get_weights()


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
