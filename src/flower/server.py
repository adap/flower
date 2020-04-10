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
from io import BytesIO
from logging import DEBUG, INFO
from typing import List, Optional, Tuple, cast

import numpy as np

from flower.client_manager import ClientManager
from flower.client_proxy import ClientProxy
from flower.history import History
from flower.logger import log
from flower.strategy import DefaultStrategy, Strategy
from flower.typing import EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Weights


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
        """Run federated averaging for a number of rounds."""
        # Initialize weights by asking one client to return theirs
        self.weights = self._get_initial_weights()
        res = self.strategy.evaluate(weights=self.weights)
        if res is not None:
            log(
                INFO, "initial weights (loss/accuracy): %s, %s", res[0], res[1],
            )

        # Run federated averaging for num_rounds
        history = History()
        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            weights_prime = self.fit_round()
            if weights_prime is not None:
                self.weights = weights_prime

            # Evaluate model using strategy implementation
            res = self.strategy.evaluate(weights=self.weights)
            if res is not None:
                log(
                    INFO,
                    "progress (round/loss/accuracy): %s, %s, %s",
                    current_round,
                    res[0],
                    res[1],
                )
                history.add_loss_centralized(rnd=current_round, loss=res[0])
                history.add_accuracy_centralized(rnd=current_round, acc=res[1])

            # Evaluate model on a sample of available clients
            if self.strategy.should_evaluate():
                loss_avg = self.evaluate()
                if loss_avg is not None:
                    history.add_loss_distributed(rnd=current_round, loss=loss_avg)

        return history

    def evaluate(self) -> Optional[float]:
        """Validate current global model on a number of clients."""
        # Sample clients for evaluation
        sample_size, min_num_clients = self.strategy.num_evaluation_clients(
            self._client_manager.num_available()
        )
        log(
            DEBUG,
            "evaluate: sample %s cids once %s clients are available",
            sample_size,
            min_num_clients,
        )
        clients = self._client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        log(
            DEBUG,
            "evaluate: sampled %s cids: %s",
            len(clients),
            [c.cid for c in clients],
        )

        # Evaluate current global weights on those clients
        parameters = weights_to_parameters(self.weights)
        evaluate_ins: FitIns = (parameters, {})
        results, failures = evaluate_clients(clients, evaluate_ins)
        log(
            DEBUG,
            "evaluate received %s results and %s failures",
            len(results),
            len(failures),
        )
        # Aggregate the evaluation results
        return self.strategy.on_aggregate_evaluate(results, failures)

    def fit_round(self) -> Optional[Weights]:
        """Perform a single round of federated averaging."""
        # Sample a number of clients (dependent on the strategy)
        sample_size, min_num_clients = self.strategy.num_fit_clients(
            self._client_manager.num_available()
        )
        log(
            DEBUG,
            "fit_round: sample %s cids once %s clients are available",
            sample_size,
            min_num_clients,
        )
        clients = self._client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        log(
            DEBUG,
            "fit_round: sampled %s cids: %s",
            len(clients),
            [c.cid for c in clients],
        )

        # Collect training results from all clients participating in this round
        parameters = weights_to_parameters(self.weights)
        fit_ins: FitIns = (parameters, {})
        results, failures = fit_clients(clients, fit_ins)
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        weights_results = [
            (parameters_to_weights(parameters), num_examples)
            for parameters, num_examples in results
        ]
        return self.strategy.on_aggregate_fit(weights_results, failures)

    def _get_initial_weights(self) -> Weights:
        """Get initial weights from one of the available clients."""
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        return parameters_to_weights(parameters_res.parameters)


def fit_clients(
    clients: List[ClientProxy], ins: FitIns
) -> Tuple[List[FitRes], List[BaseException]]:
    """Refine weights concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fit_client, c, ins) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[FitRes] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            results.append(future.result())
    return results, failures


def fit_client(client: ClientProxy, ins: FitIns) -> FitRes:
    """Refine weights on a single client."""
    return client.fit(ins)


def evaluate_clients(
    clients: List[ClientProxy], ins: EvaluateIns
) -> Tuple[List[EvaluateRes], List[BaseException]]:
    """Evaluate weights concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(evaluate_client, c, ins) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[EvaluateRes] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            results.append(future.result())
    return results, failures


def evaluate_client(client: ClientProxy, ins: EvaluateIns) -> EvaluateRes:
    """Evaluate weights on a single client."""
    return client.evaluate(ins)


def weights_to_parameters(weights: Weights) -> Parameters:
    """Convert NumPy weights to parameters object."""
    tensors = [ndarray_to_bytes(ndarray) for ndarray in weights]
    return Parameters(tensors=tensors, tensor_type="numpy.nda")


def parameters_to_weights(parameters: Parameters) -> Weights:
    """Convert parameters object to NumPy weights."""
    return [bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """Serialize NumPy array to bytes."""
    bytes_io = BytesIO()
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def bytes_to_ndarray(tensor: bytes) -> np.ndarray:
    """Deserialize NumPy array from bytes."""
    bytes_io = BytesIO(tensor)
    ndarray_deserialized = np.load(bytes_io, allow_pickle=False)
    return cast(np.ndarray, ndarray_deserialized)
