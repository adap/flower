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
"""Flower message-based FedXgbBagging strategy."""
from collections.abc import Iterable
from logging import INFO
from typing import Any, Optional, cast

import numpy as np

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.server import Grid

from .fedavg import FedAvg
from .strategy_utils import aggregate_bagging, validate_message_reply_consistency


# pylint: disable=line-too-long
class FedXgbBagging(FedAvg):
    """Configurable FedXgbBagging strategy implementation."""

    def __init__(
        self,
        **kwargs: Any,
    ):
        self.current_bst: Optional[bytes] = None
        super().__init__(**kwargs)

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        # Keep track of array record being communicated
        self.current_bst = arrays["0"].numpy().tobytes()
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        if not replies:
            return None, None

        # Log if any Messages carried errors
        # Filter messages that carry content
        num_errors = 0
        replies_with_content = []
        for msg in replies:
            if msg.has_error():
                log(
                    INFO,
                    "Received error in reply from node %d: %s",
                    msg.metadata.src_node_id,
                    msg.error,
                )
                num_errors += 1
            else:
                replies_with_content.append(msg.content)

        log(
            INFO,
            "aggregate_train: Received %s results and %s failures",
            len(replies_with_content),
            num_errors,
        )

        # Ensure expected ArrayRecords and MetricRecords are received
        validate_message_reply_consistency(
            replies=replies_with_content,
            weighted_by_key=self.weighted_by_key,
            check_arrayrecord=True,
        )

        arrays, metrics = None, None
        if replies_with_content:
            # Aggregate ArrayRecords
            for content in replies_with_content:
                bst = content["arrays"]["0"].numpy().tobytes()  # type: ignore[union-attr]
                self.current_bst = aggregate_bagging(cast(bytes, self.current_bst), bst)

            arrays = ArrayRecord(
                [np.frombuffer(cast(bytes, self.current_bst), dtype=np.uint8)]
            )

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                replies_with_content,
                self.weighted_by_key,
            )
        return arrays, metrics
