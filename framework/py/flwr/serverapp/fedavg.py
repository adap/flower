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

from logging import INFO
from typing import Callable, Optional

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

from .strategy import Strategy
from .strategy_utils import (
    aggregate_arrayrecords,
    aggregate_metricrecords,
    check_message_reply_consistency,
    sample_nodes,
)


# pylint: disable=too-many-instance-attributes
class FedAvg(Strategy):
    """Federated Averaging strategy.

    Implementation based on https://arxiv.org/abs/1602.05629

    Parameters
    ----------
    fraction_train : float, optional
        Fraction of nodes used during training. In case `min_train_nodes`
        is larger than `fraction_train * total_connected_nodes`, `min_train_nodes`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of nodes used during validation. In case `min_evaluate_nodes`
        is larger than `fraction_evaluate * total_connected_nodes`,
        `min_evaluate_nodes` will still be sampled. Defaults to 1.0.
    min_train_nodes : int, optional
        Minimum number of nodes used during training. Defaults to 2.
    min_evaluate_nodes : int, optional
        Minimum number of nodes used during validation. Defaults to 2.
    min_available_nodes : int, optional
        Minimum number of total nodes in the system. Defaults to 2.
    weighting_factor_key : str, optional
        Key used to extract the weighting factor from received MetricRecords.
        This value is used to perform weighted averaging of both ArrayRecords and
        MetricRecords. Defaults to "num-examples".
    arrayrecord_key : str, optional
        Key used to store the ArrayRecord when constructing Messages.
        Defaults to "arrays".
    configrecord_key : str, optional
         Key used to store the ConfigRecord when constructing Messages.
        Defaults to "config".
    train_metrics_aggr_fn : Callable[[list[RecordDict], str], MetricRecord], optional
        Function used to aggregate MetricRecords from training round replies.
        Takes a list of RecordDict and weighting key as input, returns aggregated
        MetricRecord.
    evaluate_metrics_aggr_fn : Callable[[list[RecordDict], str], MetricRecord], optional
        Function used to aggregate MetricRecords from evaluation round replies.
        Takes a list of RecordDict and weighting key as input, returns aggregated
        MetricRecord.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        fraction_train: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighting_factor_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: Callable[
            [list[RecordDict], str], MetricRecord
        ] = aggregate_metricrecords,
        evaluate_metrics_aggr_fn: Callable[
            [list[RecordDict], str], MetricRecord
        ] = aggregate_metricrecords,
    ) -> None:
        self.fraction_train = fraction_train
        self.fraction_evaluate = fraction_evaluate
        self.min_train_nodes = min_train_nodes
        self.min_evaluate_nodes = min_evaluate_nodes
        self.min_available_nodes = min_available_nodes
        self.weighting_factor_key = weighting_factor_key
        self.arrayrecord_key = arrayrecord_key
        self.configrecord_key = configrecord_key
        self.train_metrics_aggr_fn = train_metrics_aggr_fn
        self.evaluate_metrics_aggr_fn = evaluate_metrics_aggr_fn

    def _construct_messages(
        self, record: RecordDict, node_ids: list[int], message_type: str
    ) -> list[Message]:
        """Construct N Messages carrying the same RecordDict payload."""
        messages = []
        for node_id in node_ids:  # one message for each node
            message = Message(
                content=record,
                message_type=message_type,
                dst_node_id=node_id,
            )
            messages.append(message)
        return messages

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        """Configure the next round of federated training."""
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_train: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )
        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: list[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Log if any Messages carried errors
        num_errors = 0
        for msg in replies:
            if msg.has_error():
                log(
                    INFO,
                    "Received error in reply from node %d: %s",
                    msg.metadata.src_node_id,
                    msg.error,
                )
                num_errors += 1

        log(
            INFO,
            "aggregate_train: received %s results and %s failures",
            len(replies) - num_errors,
            num_errors,
        )

        # Filter messages that carry content
        replies_with_content = [msg.content for msg in replies if msg.has_content()]

        # Ensure expected ArrayRecords and MetricRecords are received
        skip_aggregation = check_message_reply_consistency(
            replies=replies_with_content,
            weighting_factor_key=self.weighting_factor_key,
            check_arrayrecord=True,
        )

        if skip_aggregation:
            return None, None

        # Aggregate ArrayRecords
        arrays = aggregate_arrayrecords(
            replies_with_content,
            self.weighting_factor_key,
        )

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            replies_with_content,
            self.weighting_factor_key,
        )
        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list[Message]:
        """Configure the next round of federated evaluation."""
        # Sample nodes
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_evaluate)
        sample_size = max(num_nodes, self.min_evaluate_nodes)
        node_ids, num_total = sample_nodes(grid, self.min_available_nodes, sample_size)
        log(
            INFO,
            "configure_evaluate: Sampled %s nodes (out of %s)",
            len(node_ids),
            len(num_total),
        )

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, node_ids, MessageType.EVALUATE)

    def aggregate_evaluate(
        self,
        server_round: int,
        replies: list[Message],
    ) -> Optional[MetricRecord]:
        """Aggregate MetricRecords in the received Messages."""
        # Log if any Messages carried errors
        num_errors = 0
        for msg in replies:
            if msg.has_error():
                log(
                    INFO,
                    "Received error in reply from node %d: %s",
                    msg.metadata.src_node_id,
                    msg.error,
                )
                num_errors += 1

        log(
            INFO,
            "aggregate_evaluate: received %s results and %s failures",
            len(replies) - num_errors,
            num_errors,
        )

        # Filter messages that carry content
        replies_with_content = [msg.content for msg in replies if msg.has_content()]

        # Ensure expected ArrayRecords and MetricRecords are received
        skip_aggregation = check_message_reply_consistency(
            replies=replies_with_content,
            weighting_factor_key=self.weighting_factor_key,
            check_arrayrecord=False,
        )

        if skip_aggregation:
            return None

        # Aggregate MetricRecords
        metrics = self.evaluate_metrics_aggr_fn(
            replies_with_content,
            self.weighting_factor_key,
        )
        return metrics
