# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for metric-threshold stopping through its public import path."""


from typing import Literal

import pytest

from flwr.app import MetricRecord
from flwr.serverapp.strategy import MetricThreshold, Result


@pytest.mark.parametrize(
    ("value", "comparison", "expected"),
    [
        (0.9, ">=", True),
        (0.8, ">=", False),
        (0.9, "<=", True),
        (1.0, "<=", False),
    ],
)
def test_metric_threshold(
    value: float, comparison: Literal[">=", "<="], expected: bool
) -> None:
    """Compare scalar values inclusively in either direction."""
    result = Result(evaluate_metrics_clientapp={1: MetricRecord({"accuracy": value})})
    stop_fn = MetricThreshold(
        source="evaluate_metrics_clientapp",
        metric="accuracy",
        threshold=0.9,
        comparison=comparison,
    )

    assert stop_fn(1, result) is expected


def test_missing_round_does_not_reuse_previous_metrics() -> None:
    """Continue if the current round has no record, even after a matching round."""
    stop_fn = MetricThreshold(
        source="evaluate_metrics_clientapp",
        metric="accuracy",
        threshold=0.9,
        comparison=">=",
    )
    result = Result(evaluate_metrics_clientapp={1: MetricRecord({"accuracy": 0.95})})

    assert not stop_fn(1, Result())
    assert not stop_fn(2, result)
