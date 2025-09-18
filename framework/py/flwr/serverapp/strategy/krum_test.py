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
"""Krum tests."""


from unittest.mock import MagicMock

from numpy import array, float32

from flwr.common import (
    Code,
    FitRes,
    NDArrays,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from flwr.common import ArrayRecord, RecordDict, MetricRecord
from .strategy_utils_test import create_mock_reply

from .krum import Krum


def test_aggregate_train() -> None:
    """Tests if Krum is aggregating correctly."""
    # Prepare
    strategy = Krum(
        num_malicious_nodes=1,
    )
    reply_0 = create_mock_reply(ArrayRecord(
        [array([0.2, 0.2, 0.2, 0.2], dtype=float32)]
    ), num_examples=5)
    reply_1 = create_mock_reply(ArrayRecord(
        [array([0.5, 0.5, 0.5, 0.5], dtype=float32)]
    ), num_examples=5)
    reply_2 = create_mock_reply(ArrayRecord(
        [array([1.0, 1.0, 1.0, 1.0], dtype=float32)]
    ), num_examples=5)
    replies = [reply_0, reply_1, reply_2]

    expected = ArrayRecord([array([0.7, 0.7, 0.7, 0.7], dtype=float32)])

    # Execute
    actual_aggregated, _ = strategy.aggregate_train(
        server_round=1, replies=replies
    )
    assert actual_aggregated
    actual = actual_aggregated
    assert (actual == expected[0]).all()
