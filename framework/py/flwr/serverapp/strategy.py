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
"""Flower message-based strategy."""


from abc import ABC, abstractmethod
from logging import INFO
from time import time
from typing import Callable, Optional

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.server import Grid

from .strategy_utils import ReturnStrategyResults


class Strategy(ABC):
    """Abstract base class for server strategy implementations."""

    @abstractmethod
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        """Configure the next round of training.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        arrays : ArrayRecord
            Current global ArrayRecord (e.g. global model) to be sent to client
            nodes for training.
        config : ConfigRecord
            Configuration to be sent to clients nodes for training.
        grid : Grid
            The Grid instance used for node sampling and communication.

        Returns
        -------
        list[Message]
            List of messages to be sent to selected client nodes for training.
        """

    @abstractmethod
    def aggregate_train(
        self,
        server_round: int,
        replies: list[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate training results from client nodes.

        Parameters
        ----------
        server_round : int
            The current round of federated learning, starting from 1.
        replies : list[Message]
            List of reply messages received from client nodes after training.
            Each message contains ArrayRecords and MetricRecords that get aggregated.

        Returns
        -------
        tuple[Optional[ArrayRecord], Optional[MetricRecord]]
            A tuple containing:
            - ArrayRecord: Aggregated ArrayRecord, or None if aggregation failed
            - MetricRecord: Aggregated MetricRecord, or None if aggregation failed
        """

    @abstractmethod
    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        """Configure the next round of evaluation.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        arrays : ArrayRecord
            Current global ArrayRecord (e.g. global model) to be sent to client
            nodes for evaluation.
        config : ConfigRecord
            Configuration to be sent to clients nodes for evaluation.
        grid : Grid
            The Grid instance used for node sampling and communication.

        Returns
        -------
        list[Message]
            List of messages to be sent to selected client nodes for evaluation.
        """

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        replies: list[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate evaluation metrics from client nodes.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        replies : list[Message]
            List of reply messages received from client nodes after evaluation.
            MetricRecords in the messages are aggregated.

        Returns
        -------
        Optional[MetricRecord]
            Aggregated evaluation metrics from all participating clients,
            or None if aggregation failed.
        """

    def start(
        self,
        arrays: ArrayRecord,
        grid: Grid,
        num_rounds: int,
        timeout: float,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        central_eval_fn: Optional[Callable[[int, ArrayRecord], MetricRecord]] = None,
    ) -> ReturnStrategyResults:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        num_rounds : int
            Number of federated learning rounds to execute.
        timeout : float
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        central_eval_fn : Callable[[int, ArrayRecord], MetricRecord], optional
            Optional function for centralized evaluation. Takes server round number
            and array record, returns a MetricRecord. If provided, will be called
            before the first round and after each round. Defaults to None.

        Returns
        -------
        ReturnStrategyResults
            Results containing training metrics, evaluation metrics, centralized
            evaluation metrics (if provided), and final model arrays from all rounds.
        """
        log(INFO, f"Starting {self.__class__.__name__} strategy.")
        log(INFO, f"\t└──> Number of rounds: {num_rounds}")
        log(
            INFO,
            f"\t└──> ArrayRecord: {len(arrays)} Arrays totalling "
            f"{sum(len(array.data) for array in arrays.values())/(1024**2):.2f} MB",
        )
        log(
            INFO,
            "\t└──> ConfigRecord (train): "
            f"{train_config if train_config else '(empty!)'}",
        )
        log(
            INFO,
            "\t└──> ConfigRecord (evaluate): "
            f"{evaluate_config if evaluate_config else '(empty!)'}",
        )
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        metrics_history = ReturnStrategyResults()

        t_start = time()
        # Do central eval with starting global parameters
        if central_eval_fn:
            res = central_eval_fn(0, arrays)
            log(INFO, "Initial central evaluation results: %s", res)
            metrics_history.central_evaluate_metrics[0] = res

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # Configure train, send messages and wait for replies
            replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round, arrays, train_config, grid
                ),
                timeout=timeout,
            )
            # Aggregate train
            agg_arrays, agg_metrics = self.aggregate_train(current_round, list(replies))
            # Log training metrics and append to history
            if agg_arrays is None or agg_metrics is None:
                break
            log(INFO, "\t└──> Aggregated  MetricRecord: %s", agg_metrics)
            metrics_history.train_metrics[current_round] = agg_metrics
            metrics_history.arrays = agg_arrays

            # Configure evaluate, send messages and wait for replies
            replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round, agg_arrays, evaluate_config, grid
                ),
                timeout=timeout,
            )
            # Aggregate evaluate
            agg_metrics = self.aggregate_evaluate(current_round, list(replies))
            if agg_metrics is None:
                break
            log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_metrics)
            metrics_history.evaluate_metrics[current_round] = agg_metrics

            # Centralised eval
            if central_eval_fn:
                log(INFO, "Central evaluation")
                res = central_eval_fn(current_round, agg_arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                metrics_history.central_evaluate_metrics[current_round] = res

            arrays = agg_arrays

        log(INFO, "")
        log(INFO, f"Strategy execution finished in {time() - t_start:.2f}s")
        log(INFO, "")

        return metrics_history
