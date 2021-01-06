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
import timeit
from logging import DEBUG, INFO
from typing import List, Optional, Tuple, cast

from flwr.common import (
    Disconnect,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Reconnect,
    Weights,
    parameters_to_weights,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

FitResultsAndFailures = Tuple[List[Tuple[ClientProxy, FitRes]], List[BaseException]]
EvaluateResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, EvaluateRes]], List[BaseException]
]
ReconnectResultsAndFailures = Tuple[
    List[Tuple[ClientProxy, Disconnect]], List[BaseException]
]


def set_strategy(strategy: Optional[Strategy]) -> Strategy:
    """Return Strategy."""
    return strategy if strategy is not None else FedAvg()


class Server:
    """Flower server."""

    def __init__(
        self, client_manager: ClientManager, strategy: Optional[Strategy] = None
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.weights: Weights = []
        self.strategy: Strategy = set_strategy(strategy)

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()
        # Initialize weights by asking one client to return theirs
        self.weights = self._get_initial_weights()
        res = self.strategy.evaluate(weights=self.weights)
        if res is not None:
            log(
                INFO,
                "initial weights (loss/accuracy): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(rnd=0, loss=res[0])
            history.add_accuracy_centralized(rnd=0, acc=res[1])

        # Run federated learning for num_rounds
        log(INFO, "[TIME] FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            weights_prime = self.fit_round(rnd=current_round)
            if weights_prime is not None:
                self.weights = weights_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(weights=self.weights)
            if res_cen is not None:
                loss_cen, acc_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    acc_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(rnd=current_round, loss=loss_cen)
                history.add_accuracy_centralized(rnd=current_round, acc=acc_cen)

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate(rnd=current_round)
            if res_fed is not None and res_fed[0] is not None:
                loss_fed, _ = res_fed
                history.add_loss_distributed(
                    rnd=current_round, loss=cast(float, loss_fed)
                )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "[TIME] FL finished in %s", elapsed)
        return history

    def evaluate(
        self, rnd: int
    ) -> Optional[Tuple[Optional[float], EvaluateResultsAndFailures]]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            rnd=rnd, weights=self.weights, client_manager=self._client_manager
        )
        if not client_instructions:
            log(INFO, "evaluate: no clients sampled, cancel federated evaluation")
            return None
        log(
            DEBUG,
            "evaluate: strategy sampled %s clients",
            len(client_instructions),
        )

        # Evaluate current global weights on those clients
        results_and_failures = evaluate_clients(client_instructions)
        results, failures = results_and_failures
        log(
            DEBUG,
            "evaluate received %s results and %s failures",
            len(results),
            len(failures),
        )
        # Aggregate the evaluation results
        loss_aggregated = self.strategy.aggregate_evaluate(rnd, results, failures)
        return loss_aggregated, results_and_failures

    def fit_round(self, rnd: int) -> Optional[Weights]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, weights=self.weights, client_manager=self._client_manager
        )
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )
        if not client_instructions:
            log(INFO, "fit_round: no clients sampled, cancel fit")
            return None

        # Collect training results from all clients participating in this round
        results, failures = fit_clients(client_instructions)
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        return self.strategy.aggregate_fit(rnd, results, failures)

    def disconnect_all_clients(self) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        _ = shutdown(clients=[all_clients[k] for k in all_clients.keys()])

    def _get_initial_weights(self) -> Weights:
        """Get initial weights from one of the available clients."""
        random_client = self._client_manager.sample(1)[0]
        parameters_res = random_client.get_parameters()
        return parameters_to_weights(parameters_res.parameters)


def shutdown(clients: List[ClientProxy]) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    reconnect = Reconnect(seconds=None)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(reconnect_client, c, reconnect) for c in clients]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, Disconnect]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy, reconnect: Reconnect
) -> Tuple[ClientProxy, Disconnect]:
    """Instruct a single client to disconnect and (optionally) reconnect
    later."""
    disconnect = client.reconnect(reconnect)
    return client, disconnect


def fit_clients(
    client_instructions: List[Tuple[ClientProxy, FitIns]]
) -> FitResultsAndFailures:
    """Refine weights concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(fit_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Potential success case
            result = future.result()
            if len(result[1].parameters.tensors) > 0:
                results.append(result)
            else:
                failures.append(Exception("Empty client update"))
    return results, failures


def fit_client(client: ClientProxy, ins: FitIns) -> Tuple[ClientProxy, FitRes]:
    """Refine weights on a single client."""
    fit_res = client.fit(ins)
    return client, fit_res


def evaluate_clients(
    client_instructions: List[Tuple[ClientProxy, EvaluateIns]]
) -> EvaluateResultsAndFailures:
    """Evaluate weights concurrently on all selected clients."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(evaluate_client, c, ins) for c, ins in client_instructions
        ]
        concurrent.futures.wait(futures)
    # Gather results
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    for future in futures:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            # Success case
            results.append(future.result())
    return results, failures


def evaluate_client(
    client: ClientProxy, ins: EvaluateIns
) -> Tuple[ClientProxy, EvaluateRes]:
    """Evaluate weights on a single client."""
    evaluate_res = client.evaluate(ins)
    return client, evaluate_res
