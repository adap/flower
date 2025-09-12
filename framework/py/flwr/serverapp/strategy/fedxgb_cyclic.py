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
"""Flower message-based FedXgbCyclic strategy."""


from collections.abc import Iterable
from logging import INFO
from time import sleep
from typing import Optional, cast

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

from .fedavg import FedAvg


# pylint: disable=line-too-long
class FedXgbCyclic(FedAvg):
    """Configurable FedXgbCyclic strategy implementation."""

    def _sample_nodes(
        self, grid: Grid, min_available_nodes: int, sample_size: int
    ) -> list[int]:
        """Sample all connected nodes using the Grid."""
        # Ensure min_available_nodes is at least as large as sample_size
        min_available_nodes = max(min_available_nodes, sample_size)

        # wait for min_available_nodes to be online
        while len(all_nodes := list(grid.get_node_ids())) < min_available_nodes:
            log(
                INFO,
                "Waiting for nodes to connect: %d connected (minimum required: %d).",
                len(all_nodes),
                min_available_nodes,
            )
            sleep(1)

        return all_nodes

    def _make_sampling(
        self, grid: Grid, server_round: int, configure_type: str
    ) -> list[int]:
        """Sample nodes using the Grid."""
        num_nodes = int(len(list(grid.get_node_ids())) * self.fraction_train)
        sample_size = max(num_nodes, self.min_train_nodes)
        node_ids = self._sample_nodes(grid, self.min_available_nodes, sample_size)

        # Sample the clients sequentially given server_round
        sampled_idx = server_round % len(node_ids)
        sampled_node_id = [node_ids[sampled_idx]]

        log(
            INFO,
            f"{configure_type}: Sampled %s nodes (out of %s)",
            len(sampled_node_id),
            len(node_ids),
        )
        return sampled_node_id

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Sample nodes
        sampled_node_id = self._make_sampling(grid, server_round, "configure_train")

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, sampled_node_id, MessageType.TRAIN)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        arrays, metrics = None, None
        if valid_replies:
            reply_contents = [msg.content for msg in valid_replies]

            # Fetch the client model from last round as global model
            arrays = cast(ArrayRecord, reply_contents[0]["arrays"])

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return arrays, metrics

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated evaluation."""
        # Sample nodes
        sampled_node_id = self._make_sampling(grid, server_round, "configure_evaluate")

        # Always inject current server round
        config["server-round"] = server_round

        # Construct messages
        record = RecordDict(
            {self.arrayrecord_key: arrays, self.configrecord_key: config}
        )
        return self._construct_messages(record, sampled_node_id, MessageType.EVALUATE)
