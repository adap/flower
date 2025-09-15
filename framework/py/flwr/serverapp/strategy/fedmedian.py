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
"""Federated Median (FedMedian) [Yin et al., 2018] strategy.

Paper: arxiv.org/pdf/1803.01498v1.pdf
"""


from collections.abc import Iterable
from typing import Optional, cast

import numpy as np

from flwr.common import Array, ArrayRecord, Message, MetricRecord

from .fedavg import FedAvg


class FedMedian(FedAvg):
    """Federated Median (FedMedian) strategy.

    Implementation based on https://arxiv.org/pdf/1803.01498v1
    """

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Call FedAvg aggregate_train to perform validation and aggregation
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)

        if not valid_replies:
            return None, None

        # Aggregate ArrayRecords using median
        # Get the key for the only ArrayRecord from the first Message
        record_key = list(valid_replies[0].content.array_records.keys())[0]
        # Preserve keys for arrays in ArrayRecord
        array_keys = list(valid_replies[0].content[record_key].keys())

        # Compute median for each layer and construct ArrayRecord
        arrays = ArrayRecord()
        for array_key in array_keys:
            # Get the corresponding layer from each client
            layers = [
                cast(ArrayRecord, msg.content[record_key]).pop(array_key).numpy()
                for msg in valid_replies
            ]
            # Compute median and save as Array in ArrayRecord
            arrays[array_key] = Array(np.median(np.stack(layers), axis=0))

        # Aggregate MetricRecords
        metrics = self.train_metrics_aggr_fn(
            [msg.content for msg in valid_replies],
            self.weighted_by_key,
        )
        return arrays, metrics
