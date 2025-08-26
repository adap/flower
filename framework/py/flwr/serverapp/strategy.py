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


import time
from abc import ABC, abstractmethod
from logging import INFO
from typing import Callable, Optional

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.server import Grid

from .result import Result
from .strategy_utils import log_strategy_start_info


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

    @abstractmethod
    def summary(self) -> None:
        """Log summary configuration of the strategy."""

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int,
        timeout: float,
        train_config: Optional[ConfigRecord] = None,
        evaluate_config: Optional[ConfigRecord] = None,
        central_eval_fn: Optional[Callable[[int, ArrayRecord], MetricRecord]] = None,
    ) -> Result:
        """Execute the federated learning strategy.

        Runs the complete federated learning workflow for the specified number of
        rounds, including training, evaluation, and optional centralized evaluation.

        Parameters
        ----------
        grid : Grid
            The Grid instance used to send/receive Messages from nodes executing a
            ClientApp.
        initial_arrays : ArrayRecord
            Initial model parameters (arrays) to be used for federated learning.
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
        StrategyResults
            Results containing training metrics, evaluation metrics, centralized
            evaluation metrics (if provided), and final model arrays from all rounds.
        """
        log(INFO, "Starting %s strategy:", self.__class__.__name__)
        log_strategy_start_info(
            num_rounds, initial_arrays, train_config, evaluate_config
        )
        self.summary()
        log(INFO, "")

        # Initialize if None
        train_config = ConfigRecord() if train_config is None else train_config
        evaluate_config = ConfigRecord() if evaluate_config is None else evaluate_config
        results = Result()

        t_start = time.time()
        # Do central eval with starting global parameters
        if central_eval_fn:
            res = central_eval_fn(0, initial_arrays)
            log(INFO, "Initial central evaluation results: %s", res)
            results.central_evaluate_metrics[0] = res

        arrays = initial_arrays
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
            replies_list = list(replies)
            if replies_list:
                # Aggregate train
                agg_arrays, agg_metrics = self.aggregate_train(
                    current_round, replies_list
                )
                del replies_list
                # Log training metrics and append to history
                if agg_arrays is None or agg_metrics is None:
                    break
                log(INFO, "\t└──> Aggregated  MetricRecord: %s", agg_metrics)
                results.train_metrics[current_round] = agg_metrics
                results.arrays = agg_arrays
                arrays = agg_arrays

            # Configure evaluate, send messages and wait for replies
            replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round, arrays, evaluate_config, grid
                ),
                timeout=timeout,
            )
            replies_list = list(replies)
            if replies_list:
                # Aggregate evaluate
                agg_metrics = self.aggregate_evaluate(current_round, replies_list)
                if agg_metrics is None:
                    break
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_metrics)
                results.evaluate_metrics[current_round] = agg_metrics

            # Centralised eval
            if central_eval_fn:
                log(INFO, "Central evaluation")
                res = central_eval_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                results.central_evaluate_metrics[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")

        return results
