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
"""Metric-based stopping predicate for message-based strategies."""


from math import isfinite
from typing import Literal

from .result import Result


class MetricThreshold:
    """Stop when a current-round metric reaches a threshold.

    Pass an instance as ``stop_fn`` to
    :meth:`~flwr.serverapp.strategy.Strategy.start`. The comparison is inclusive.
    If the selected source has no record for the current round, return
    False without consulting previous rounds. A missing key in an existing record
    or a value that is not a finite scalar raises ValueError.

    Parameters
    ----------
    source : str
        Name of the metric history in Result to monitor: "train_metrics_clientapp",
        "evaluate_metrics_clientapp", or "evaluate_metrics_serverapp".
    metric : str
        Key of the scalar metric to monitor in the selected MetricRecord.
    threshold : float
        Finite target value for the metric.
    comparison : {">=", "<="}
        Stop when the metric is greater than or equal to the threshold (">=") or
        less than or equal to the threshold ("<=").

    Examples
    --------
    >>> from flwr.serverapp.strategy import MetricThreshold
    >>> stop_fn = MetricThreshold(
    ...     source="evaluate_metrics_clientapp",
    ...     metric="accuracy",
    ...     threshold=0.9,
    ...     comparison=">=",
    ... )
    """

    def __init__(
        self,
        *,
        source: Literal[
            "train_metrics_clientapp",
            "evaluate_metrics_clientapp",
            "evaluate_metrics_serverapp",
        ],
        metric: str,
        threshold: float,
        comparison: Literal[">=", "<="],
    ) -> None:
        if source not in (
            "train_metrics_clientapp",
            "evaluate_metrics_clientapp",
            "evaluate_metrics_serverapp",
        ):
            raise ValueError(f"Unknown metric source: {source!r}.")
        if comparison not in (">=", "<="):
            raise ValueError("comparison must be '>=' or '<='.")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or (isinstance(threshold, float) and not isfinite(threshold))
        ):
            raise ValueError("threshold must be a finite scalar number.")

        self.source = source
        self.metric = metric
        self.threshold = threshold
        self.comparison = comparison

    def __call__(self, server_round: int, result: Result) -> bool:
        """Check the selected metric from the current round.

        Parameters
        ----------
        server_round : int
            Current completed round number.
        result : Result
            Accumulated strategy results.

        Returns
        -------
        bool
            Whether the selected metric reaches the threshold.
        """
        history = {
            "train_metrics_clientapp": result.train_metrics_clientapp,
            "evaluate_metrics_clientapp": result.evaluate_metrics_clientapp,
            "evaluate_metrics_serverapp": result.evaluate_metrics_serverapp,
        }[self.source]
        metrics = history.get(server_round)
        if metrics is None:
            return False
        if self.metric not in metrics:
            raise ValueError(
                f"Metric {self.metric!r} not found in {self.source} "
                f"for round {server_round}."
            )
        value = metrics[self.metric]
        if not isinstance(value, (int, float)) or (
            isinstance(value, float) and not isfinite(value)
        ):
            raise ValueError(f"Metric {self.metric!r} must be a finite scalar number.")
        if self.comparison == ">=":
            return value >= self.threshold
        return value <= self.threshold
