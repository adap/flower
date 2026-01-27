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
"""Flower message-based FedAvg strategy."""


from collections.abc import Callable, Iterable
from logging import INFO, WARNING
import time

from flwr.common import (
    ArrayRecord,
    ConfigRecord,
    Message,
    MessageType,
    MetricRecord,
    RecordDict,
    log,
)
from flwr.server import Grid

from flwr.serverapp.strategy import FedAvg, Result
from flwr.serverapp.strategy.strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    sample_nodes,
    validate_message_reply_consistency,
)


# pylint: disable=too-many-instance-attributes
class FedAvgStreaming(FedAvg):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_train : float (default: 1.0)
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled.
    fraction_evaluate : float (default: 1.0)
        Fraction of nodes used during validation. In case `min_evaluate_nodes`
        is larger than `fraction_evaluate * total_connected_nodes`,
        `min_evaluate_nodes` will still be sampled.
    min_train_nodes : int (default: 2)
        Minimum number of nodes used during training.
    min_evaluate_nodes : int (default: 2)
        Minimum number of nodes used during validation.
    min_available_nodes : int (default: 2)
        Minimum number of total nodes in the system.
    weighted_by_key : str (default: "num-examples")
        The key within each MetricRecord whose value is used as the weight when
        computing weighted averages for both ArrayRecords and MetricRecords.
    arrayrecord_key : str (default: "arrays")
        Key used to store the ArrayRecord when constructing Messages.
    configrecord_key : str (default: "config")
        Key used to store the ConfigRecord when constructing Messages.
    train_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    evaluate_metrics_aggr_fn : Optional[callable] (default: None)
        Function with signature (list[RecordDict], str) -> MetricRecord,
        used to aggregate MetricRecords from training round replies.
        If `None`, defaults to `aggregate_metricrecords`, which performs a weighted
        average using the provided weight factor key.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
        evaluate_metrics_aggr_fn: (
            Callable[[list[RecordDict], str], MetricRecord] | None
        ) = None,
    ) -> None:
        super().__init__(
            fraction_train = fraction_train,
            fraction_evaluate = fraction_evaluate,
            min_train_nodes = min_train_nodes,
            min_evaluate_nodes = min_evaluate_nodes,
            min_available_nodes = min_available_nodes,
            weighted_by_key = weighted_by_key,
            arrayrecord_key = arrayrecord_key,
            configrecord_key = configrecord_key,
            train_metrics_aggr_fn = train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn = evaluate_metrics_aggr_fn,
        )

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            print(f'Aggregating models...')   
            start_time = time.time()
            # Aggregate ArrayRecords
            arrays = aggregate_arrayrecords(
                reply_contents,
                self.weighted_by_key,
            )
            end_time = time.time()
            print(f'Took {end_time - start_time} seconds to aggregate models')

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return arrays, metrics

    # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    def start(
        self,
        grid: Grid,
        initial_arrays: ArrayRecord,
        num_rounds: int = 3,
        timeout: float = 3600,
        train_config: ConfigRecord | None = None,
        evaluate_config: ConfigRecord | None = None,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord | None] | None = None,
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
        num_rounds : int (default: 3)
            Number of federated learning rounds to execute.
        timeout : float (default: 3600)
            Timeout in seconds for waiting for node responses.
        train_config : ConfigRecord, optional
            Configuration to be sent to nodes during training rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_config : ConfigRecord, optional
            Configuration to be sent to nodes during evaluation rounds.
            If unset, an empty ConfigRecord will be used.
        evaluate_fn : Callable[[int, ArrayRecord], Optional[MetricRecord]], optional
            Optional function for centralized evaluation of the global model. Takes
            server round number and array record, returns a MetricRecord or None. If
            provided, will be called before the first round and after each round.
            Defaults to None.

        Returns
        -------
        Results
            Results containing final model arrays and also training metrics, evaluation
            metrics and global evaluation metrics (if provided) from all rounds.
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
        result = Result()

        t_start = time.time()
        # Evaluate starting global parameters
        if evaluate_fn:
            res = evaluate_fn(0, initial_arrays)
            log(INFO, "Initial global evaluation results: %s", res)
            if res is not None:
                result.evaluate_metrics_serverapp[0] = res

        arrays = initial_arrays

        for current_round in range(1, num_rounds + 1):
            log(INFO, "")
            log(INFO, "[ROUND %s/%s]", current_round, num_rounds)

            # -----------------------------------------------------------------
            # --- TRAINING (CLIENTAPP-SIDE) -----------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure training round
            # Send messages and wait for replies
            train_replies = grid.send_and_receive(
                messages=self.configure_train(
                    current_round,
                    arrays,
                    train_config,
                    grid,
                ),
                timeout=timeout,
            )

            send_complete = False
            while(not send_complete):
                print(f'Requesting layer...')
                # Call strategy to configure training round
                # Send messages and wait for replies
                node_ids = [msg.metadata.src_node_id for msg in train_replies]
                train_comms_replies = grid.send_and_receive(
                    messages=self._construct_messages(
                        RecordDict(),
                        node_ids,
                        message_type="train.layer_wise_communication"
                    )
                )
                train_comms_replies = [msg.content for msg in train_comms_replies]
                


      

            # Aggregate train
            agg_arrays, agg_train_metrics = self.aggregate_train(
                current_round,
                train_replies,
            )

            # Log training metrics and append to history
            if agg_arrays is not None:
                result.arrays = agg_arrays
                arrays = agg_arrays
            if agg_train_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_train_metrics)
                result.train_metrics_clientapp[current_round] = agg_train_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (CLIENTAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Call strategy to configure evaluation round
            # Send messages and wait for replies
            evaluate_replies = grid.send_and_receive(
                messages=self.configure_evaluate(
                    current_round,
                    arrays,
                    evaluate_config,
                    grid,
                ),
                timeout=timeout,
            )

            # Aggregate evaluate
            agg_evaluate_metrics = self.aggregate_evaluate(
                current_round,
                evaluate_replies,
            )

            # Log training metrics and append to history
            if agg_evaluate_metrics is not None:
                log(INFO, "\t└──> Aggregated MetricRecord: %s", agg_evaluate_metrics)
                result.evaluate_metrics_clientapp[current_round] = agg_evaluate_metrics

            # -----------------------------------------------------------------
            # --- EVALUATION (SERVERAPP-SIDE) ---------------------------------
            # -----------------------------------------------------------------

            # Centralized evaluation
            if evaluate_fn:
                log(INFO, "Global evaluation")
                res = evaluate_fn(current_round, arrays)
                log(INFO, "\t└──> MetricRecord: %s", res)
                if res is not None:
                    result.evaluate_metrics_serverapp[current_round] = res

        log(INFO, "")
        log(INFO, "Strategy execution finished in %.2fs", time.time() - t_start)
        log(INFO, "")
        log(INFO, "Final results:")
        log(INFO, "")
        for line in io.StringIO(str(result)):
            log(INFO, "\t%s", line.strip("\n"))
        log(INFO, "")

        return result  
