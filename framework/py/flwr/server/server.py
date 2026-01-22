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
"""Flower server."""


import concurrent.futures
import threading
import io
import os
import timeit
from logging import INFO, WARN
from typing import Callable, Optional, TypeVar, Union

from flwr.common import (
    Code,
    DisconnectRes,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    ReconnectIns,
    Scalar,
)


TResult = TypeVar("TResult")


class _ConcurrencyTracker:
    """Track current and peak concurrency for a batch of tasks."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.current = 0
        self.peak = 0

    def increment(self) -> None:
        with self._lock:
            self.current += 1
            if self.current > self.peak:
                self.peak = self.current

    def decrement(self) -> None:
        with self._lock:
            self.current -= 1


def _track_concurrency(
    tracker: _ConcurrencyTracker, func: Callable[..., TResult], *args: object
) -> TResult:
    tracker.increment()
    try:
        return func(*args)
    finally:
        tracker.decrement()
from flwr.common.logger import log
from flwr.common.typing import GetParametersIns
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from .server_config import ServerConfig
from ..common.crypto import log_file
from ..common.crypto.log_file import log_time

FitResultsAndFailures = tuple[
    list[tuple[ClientProxy, FitRes]],
    list[Union[tuple[ClientProxy, FitRes], BaseException]],
]
EvaluateResultsAndFailures = tuple[
    list[tuple[ClientProxy, EvaluateRes]],
    list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
]
ReconnectResultsAndFailures = tuple[
    list[tuple[ClientProxy, DisconnectRes]],
    list[Union[tuple[ClientProxy, DisconnectRes], BaseException]],
]


class Server:
    """Flower server."""

    def __init__(
        self,
        *,
        client_manager: ClientManager,
        strategy: Optional[Strategy] = None,
    ) -> None:
        self._client_manager: ClientManager = client_manager
        self.parameters: Parameters = Parameters(
            tensors=[], tensor_type="numpy.ndarray"
        )
        self.strategy: Strategy = strategy if strategy is not None else FedAvg()
        self.max_workers: Optional[int] = None

    def set_max_workers(self, max_workers: Optional[int]) -> None:
        """Set the max_workers used by ThreadPoolExecutor."""
        self.max_workers = max_workers

    def set_strategy(self, strategy: Strategy) -> None:
        """Replace server strategy."""
        self.strategy = strategy

    def client_manager(self) -> ClientManager:
        """Return ClientManager."""
        return self._client_manager

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> tuple[History, float]:
        """Run federated averaging for a number of rounds."""
        log(INFO, "inizio1")
        start_time = timeit.default_timer()
        log_file.reset_crypto_totals()
        history = History()
        if hasattr(self.strategy, "history"):
                self.strategy.history = history
        num_clients = self._client_manager.num_available();
        log_time("Numero totale di client disponibili: %s", num_clients)
        # Initialize parameters
        log(INFO, "[INIT]")
        self.parameters = self._get_initial_parameters(server_round=0, timeout=timeout)
        log(INFO, "Starting evaluation of initial global parameters")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])
        else:
            log(INFO, "Evaluation returned no results (`None`)")

        # Run federated learning for num_rounds
        prev_crypto_total, _ = log_file.get_crypto_totals()

        for current_round in range(1, num_rounds + 1):
            if getattr(self.strategy, "stop_triggered", False):
                log(INFO, "Early stopping triggered at round %s, stopping server.", current_round - 1)
                log_time("Early stopping triggered at round %s, stopping server.", current_round - 1)
                break

            round_start = timeit.default_timer()
            log(INFO, "[ROUND %s]", current_round)
            log_time(f"[ROUND {current_round}]")

            # Train model
            round_fit_clients = 0
            round_eval_clients = 0
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            round_fit_parallel = 0
            if res_fit is not None:
                (
                    parameters_prime,
                    fit_metrics,
                    (fit_results, _),
                    round_fit_parallel,
                ) = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime
                history.add_metrics_distributed_fit(
                    server_round=current_round, metrics=fit_metrics
                )
                round_fit_clients = len(fit_results)

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(INFO, "fit progress: (%s, %s, %s, %s)",
                    current_round, loss_cen, metrics_cen, timeit.default_timer() - round_start)
                log_time(f"fit progress: ({current_round}, {loss_cen}, {metrics_cen}, {timeit.default_timer() - round_start:.5f}s)")
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )
                print("metrics_cen",metrics_cen )
                if "accuracy" in metrics_cen:
                    log_time("Round %s Accuracy (centralized): %.4f", current_round, metrics_cen["accuracy"])

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            round_eval_parallel = 0
            if res_fed is not None:
                (
                    loss_fed,
                    evaluate_metrics_fed,
                    (eval_results, _),
                    round_eval_parallel,
                ) = res_fed
                if loss_fed is not None:
                    history.add_loss_distributed(server_round=current_round, loss=loss_fed)
                    history.add_metrics_distributed(server_round=current_round, metrics=evaluate_metrics_fed)
                    if "accuracy" in evaluate_metrics_fed:
                       log_time("Round %s Accuracy (federated): %.4f", current_round, evaluate_metrics_fed["accuracy"])
                round_eval_clients = len(eval_results)
            # Fine round: calcolo e log del tempo
            round_elapsed = timeit.default_timer() - round_start
            current_crypto_total, _ = log_file.get_crypto_totals()
            round_crypto_time = max(current_crypto_total - prev_crypto_total, 0.0)
            prev_crypto_total = current_crypto_total
            parallel_factor = max(round_fit_parallel, round_eval_parallel, 1)
            parallel_crypto_time = min(
                round_crypto_time / parallel_factor, round_elapsed
            )
            without_crypto = max(round_elapsed - parallel_crypto_time, 0.0)
            log_file.ROUND_SUMMARIES.append({
                "round": current_round,
                "round_time": round_elapsed,
                "crypto_time": parallel_crypto_time,
                "crypto_cumulative": round_crypto_time,
                "parallel_fit": float(round_fit_parallel),
                "parallel_eval": float(round_eval_parallel),
                "parallel_factor": float(parallel_factor),
                "without_crypto": without_crypto,
            })

            log_time("Tempo totale round %s: %.2f s", current_round, round_elapsed)

            history.add_metrics_centralized(
                server_round=current_round,
                metrics={"round_time": round_elapsed}
            )

        # Bookkeeping
       #. if log_file.is_report_requested():
            log_time("=== Report tempi round ===")
            for line in log_file.build_round_time_report():
                log_time(line)
        end_time = timeit.default_timer()

        elapsed= end_time - start_time
        return history, elapsed

    def evaluate_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        tuple[Optional[float], dict[str, Scalar], EvaluateResultsAndFailures, int]
    ]:
        """Validate current global model on a number of clients."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "configure_evaluate: no clients selected, skipping evaluation")
            return None
        log(
            INFO,
            "configure_evaluate: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        (results, failures), eval_parallel = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: tuple[
            Optional[float],
            dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures), eval_parallel

    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        tuple[Optional[Parameters], dict[str, Scalar], FitResultsAndFailures, int]
    ]:
        """Perform a single round of federated averaging."""
        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        (results, failures), fit_parallel = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )

        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures), fit_parallel

    def disconnect_all_clients(self, timeout: Optional[float]) -> None:
        """Send shutdown signal to all clients."""
        all_clients = self._client_manager.all()
        clients = [all_clients[k] for k in all_clients.keys()]
        instruction = ReconnectIns(seconds=None)
        client_instructions = [(client_proxy, instruction) for client_proxy in clients]
        _ = reconnect_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )

    def _get_initial_parameters(
        self, server_round: int, timeout: Optional[float]
    ) -> Parameters:
        """Get initial parameters from one of the available clients."""
        # Server-side parameter initialization
        parameters: Optional[Parameters] = self.strategy.initialize_parameters(
            client_manager=self._client_manager
        )
        if parameters is not None:
            log(INFO, "Using initial global parameters provided by strategy")
            return parameters

        # Get initial parameters from one of the clients
        log(INFO, "Requesting initial parameters from one random client")
        random_client = self._client_manager.sample(1)[0]
        ins = GetParametersIns(config={})
        get_parameters_res = random_client.get_parameters(
            ins=ins, timeout=timeout, group_id=server_round
        )
        if get_parameters_res.status.code == Code.OK:
            log(INFO, "Received initial parameters from one random client")
        else:
            log(
                WARN,
                "Failed to receive initial parameters from the client."
                " Empty initial parameters will be used.",
            )
        return get_parameters_res.parameters


def reconnect_clients(
    client_instructions: list[tuple[ClientProxy, ReconnectIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
) -> ReconnectResultsAndFailures:
    """Instruct clients to disconnect and never reconnect."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(reconnect_client, client_proxy, ins, timeout)
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, DisconnectRes]] = []
    failures: list[Union[tuple[ClientProxy, DisconnectRes], BaseException]] = []
    for future in finished_fs:
        failure = future.exception()
        if failure is not None:
            failures.append(failure)
        else:
            result = future.result()
            results.append(result)
    return results, failures


def reconnect_client(
    client: ClientProxy,
    reconnect: ReconnectIns,
    timeout: Optional[float],
) -> tuple[ClientProxy, DisconnectRes]:
    """Instruct client to disconnect and (optionally) reconnect later."""
    disconnect = client.reconnect(
        reconnect,
        timeout=timeout,
        group_id=None,
    )
    return client, disconnect


def fit_clients(
    client_instructions: list[tuple[ClientProxy, FitIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    group_id: int,
) -> tuple[FitResultsAndFailures, int]:
    """Refine parameters concurrently on all selected clients."""
    tracker = _ConcurrencyTracker()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(
                _track_concurrency,
                tracker,
                fit_client,
                client_proxy,
                ins,
                timeout,
                group_id,
            )
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, FitRes]] = []
    failures: list[Union[tuple[ClientProxy, FitRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_fit(
            future=future, results=results, failures=failures
        )
    return (results, failures), tracker.peak


def fit_client(
    client: ClientProxy, ins: FitIns, timeout: Optional[float], group_id: int
) -> tuple[ClientProxy, FitRes]:
    """Refine parameters on a single client."""
    fit_res = client.fit(ins, timeout=timeout, group_id=group_id)
    return client, fit_res


def _handle_finished_future_after_fit(
    future: concurrent.futures.Future,  # type: ignore
    results: list[tuple[ClientProxy, FitRes]],
    failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, FitRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def evaluate_clients(
    client_instructions: list[tuple[ClientProxy, EvaluateIns]],
    max_workers: Optional[int],
    timeout: Optional[float],
    group_id: int,
) -> tuple[EvaluateResultsAndFailures, int]:
    """Evaluate parameters concurrently on all selected clients."""
    tracker = _ConcurrencyTracker()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(
                _track_concurrency,
                tracker,
                evaluate_client,
                client_proxy,
                ins,
                timeout,
                group_id,
            )
            for client_proxy, ins in client_instructions
        }
        finished_fs, _ = concurrent.futures.wait(
            fs=submitted_fs,
            timeout=None,  # Handled in the respective communication stack
        )

    # Gather results
    results: list[tuple[ClientProxy, EvaluateRes]] = []
    failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]] = []
    for future in finished_fs:
        _handle_finished_future_after_evaluate(
            future=future, results=results, failures=failures
        )
    return (results, failures), tracker.peak


def evaluate_client(
    client: ClientProxy,
    ins: EvaluateIns,
    timeout: Optional[float],
    group_id: int,
) -> tuple[ClientProxy, EvaluateRes]:
    """Evaluate parameters on a single client."""
    evaluate_res = client.evaluate(ins, timeout=timeout, group_id=group_id)
    return client, evaluate_res


def _handle_finished_future_after_evaluate(
    future: concurrent.futures.Future,  # type: ignore
    results: list[tuple[ClientProxy, EvaluateRes]],
    failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, EvaluateRes] = future.result()
    _, res = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append(result)
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)


def init_defaults(
    server: Optional[Server],
    config: Optional[ServerConfig],
    strategy: Optional[Strategy],
    client_manager: Optional[ClientManager],
) -> tuple[Server, ServerConfig]:
    """Create server instance if none was given."""
    if server is None:
        if client_manager is None:
            client_manager = SimpleClientManager()
        if strategy is None:
            strategy = FedAvg()
        server = Server(client_manager=client_manager, strategy=strategy)
    elif strategy is not None:
        log(WARN, "Both server and strategy were provided, ignoring strategy")

    # Set default config values
    if config is None:
        config = ServerConfig()

    return server, config

import requests

BOT_TOKEN = "8440783074:AAGBenk_eeglVRWIIvuNACUBCkhSxVJoAio"
CHAT_ID = 587180276

def send_telegram_file(file_path: str, caption: str = ""):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            files = {"document": f}
            data = {"chat_id": CHAT_ID, "caption": caption}
            requests.post(url, data=data, files=files)
    except Exception as e:
        print(f"Errore invio file Telegram: {e}")

def run_fl(
        server: Server,
        config: ServerConfig,
) -> History:
    """Train a model on the given server and return the History object."""
    hist, elapsed_time = server.fit(
        num_rounds=config.num_rounds, timeout=config.round_timeout
    )

    log(INFO, "")
    log(INFO, "[SUMMARY]")
    log(INFO, "Run finished %s round(s) in %.2fs", config.num_rounds, elapsed_time)
    log_time("Run finished %s round(s) in %.2fs", config.num_rounds, elapsed_time)
    total_crypto_time, total_serial_time = log_file.get_crypto_totals()
    crypto_impact = (
        (total_crypto_time / elapsed_time * 100.0) if elapsed_time > 0 else 0.0
    )
    log_time(
        "Totale critto: %.2f s su %.2f s (%.2f%%) | serializzazione: %.2f s",
        total_crypto_time,
        elapsed_time,
        crypto_impact,
        total_serial_time,
    )

    # 📩 Messaggio Telegram
    #send_telegram_file(null,"Ho finito!!!!")

    # 📄 Invio CSV se esiste


    for line in io.StringIO(str(hist)):
        log_time("\t%s", line.strip("\n"))
        log_time("")

    log_time("")  # riga vuota finale
    if log_file.CSV_PATH is not None:
        abs_path = os.path.abspath(log_file.CSV_PATH)

        if os.path.exists(abs_path):
            send_telegram_file(abs_path, caption="Ecco il log del run")
        else:
            print(f"[run_fl] File non trovato: {abs_path}")
    else:
        print("[run_fl] Nessun CSV_PATH disponibile")
    # Graceful shutdown
    server.disconnect_all_clients(timeout=config.round_timeout)

    return hist
