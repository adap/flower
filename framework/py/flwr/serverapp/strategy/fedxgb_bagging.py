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
from typing import Optional, cast

import numpy as np

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.server import Grid

from ..exception import InconsistentMessageReplies
from .fedavg import FedAvg
from .strategy_utils import aggregate_bagging


# pylint: disable=line-too-long
class FedXgbBagging(FedAvg):
    """Configurable FedXgbBagging strategy implementation."""

    current_bst: Optional[bytes] = None

    def _ensure_single_array(self, arrays: ArrayRecord) -> None:
        """Check that ensures there's only one Array in the ArrayRecord."""
        n = len(arrays)
        if n != 1:
            raise InconsistentMessageReplies(
                reason="Expected exactly one Array in ArrayRecord. "
                "Skipping aggregation."
            )

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of federated training."""
        self._ensure_single_array(arrays)
        # Keep track of array record being communicated
        self.current_bst = arrays["0"].numpy().tobytes()
        return super().configure_train(server_round, arrays, config, grid)

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
            array_record_key = next(iter(reply_contents[0].array_records.keys()))

            # Aggregate ArrayRecords
            for content in reply_contents:
                self._ensure_single_array(cast(ArrayRecord, content[array_record_key]))
                bst = content[array_record_key]["0"].numpy().tobytes()  # type: ignore[union-attr]

                if self.current_bst is not None:
                    self.current_bst = aggregate_bagging(self.current_bst, bst)

            if self.current_bst is not None:
                arrays = ArrayRecord([np.frombuffer(self.current_bst, dtype=np.uint8)])

            # Aggregate MetricRecords
            metrics = self.train_metrics_aggr_fn(
                reply_contents,
                self.weighted_by_key,
            )
        return arrays, metrics
