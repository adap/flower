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
"""Flower message-based strategy utilities."""


import random
from collections import OrderedDict
from dataclasses import dataclass, field
from logging import ERROR, INFO
from time import sleep
from typing import Optional, cast

import numpy as np

from flwr.common import (
    Array,
    ArrayRecord,
    ConfigRecord,
    MetricRecord,
    NDArray,
    RecordDict,
    log,
)
from flwr.server import Grid


@dataclass
class StrategyResults:
    """Data class carrying records generated during the execution of a strategy."""

    arrays: ArrayRecord = field(default_factory=ArrayRecord)
    train_metrics: dict[int, MetricRecord] = field(default_factory=dict)
    evaluate_metrics: dict[int, MetricRecord] = field(default_factory=dict)
    central_evaluate_metrics: dict[int, MetricRecord] = field(default_factory=dict)


def config_to_str(configRecord: ConfigRecord) -> str:
    """Convert a ConfigRecord to a string representation masking bytes."""
    content = ", ".join(
        f"'{k}': {'<bytes>' if isinstance(v, bytes) else v}"
        for k, v in configRecord.items()
    )
    return f"{{{content}}}"


def log_strategy_start_info(
    num_rounds: int,
    arrays: ArrayRecord,
    train_config: Optional[ConfigRecord],
    evaluate_config: Optional[ConfigRecord],
) -> None:
    """Log information about the strategy start."""
    log(INFO, f"\t└──> Number of rounds: {num_rounds}")
    log(
        INFO,
        f"\t└──> ArrayRecord: {len(arrays)} Arrays totalling "
        f"{sum(len(array.data) for array in arrays.values())/(1024**2):.2f} MB",
    )
    log(
        INFO,
        "\t└──> ConfigRecord (train): "
        f"{config_to_str(train_config) if train_config else '(empty!)'}",
    )
    log(
        INFO,
        "\t└──> ConfigRecord (evaluate): "
        f"{config_to_str(evaluate_config) if evaluate_config else '(empty!)'}",
    )
    log(INFO, "")


def aggregate_arrayrecords(
    records: list[RecordDict], weighting_metric_name: str
) -> ArrayRecord:
    """Perform weighted aggregation all ArrayRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        # Because replies have been checked for consistency,
        # we can safely cast the weighting factor to float
        w = cast(float, metricrecord[weighting_metric_name])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    # Perform weighted aggregation
    aggregated_np_arrays: dict[str, NDArray] = {}

    for record, weight in zip(records, weight_factors):
        for record_item in record.array_records.values():
            # aggregate in-place
            for key, value in record_item.items():
                if key not in aggregated_np_arrays:
                    aggregated_np_arrays[key] = value.numpy() * weight
                else:
                    aggregated_np_arrays[key] += value.numpy() * weight

    return ArrayRecord(
        OrderedDict({k: Array(v) for k, v in aggregated_np_arrays.items()})
    )


def aggregate_metricrecords(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Perform weighted aggregation all MetricRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        # Because replies have been checked for consistency,
        # we can safely cast the weighting factor to float
        w = cast(float, metricrecord[weighting_metric_name])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    aggregated_metrics = MetricRecord()
    for record, weight in zip(records, weight_factors):
        for record_item in record.metric_records.values():
            # aggregate in-place
            for key, value in record_item.items():
                if key == weighting_metric_name:
                    # We exclude the weighting key from the aggregated MetricRecord
                    continue
                if key not in aggregated_metrics:
                    if isinstance(value, list):
                        aggregated_metrics[key] = (np.array(value) * weight).tolist()
                    else:
                        aggregated_metrics[key] = value * weight
                else:
                    if isinstance(value, list):
                        aggregated_metrics[key] = (
                            np.array(aggregated_metrics[key]) + np.array(value) * weight
                        ).tolist()
                    else:
                        current_value = cast(float, aggregated_metrics[key])
                        aggregated_metrics[key] = current_value + value * weight

    return aggregated_metrics


def sample_nodes(
    grid: Grid, min_available_nodes: int, sample_size: int
) -> tuple[list[int], list[int]]:
    """Sample the specified number of nodes using the Grid."""
    sampled_nodes = []

    # wait for min_available_nodes to be online
    nodes_connected = list(grid.get_node_ids())
    while len(nodes_connected) < min_available_nodes:
        sleep(1)
        log(
            INFO,
            f"Waiting for nodes to connect. Nodes connected {len(nodes_connected)} "
            f"(expecting at least {min_available_nodes}).",
        )
        nodes_connected = list(grid.get_node_ids())

    # Sample nodes
    sampled_nodes = random.sample(list(nodes_connected), sample_size)

    return sampled_nodes, nodes_connected


def check_message_reply_consistency(
    replies: list[RecordDict], weighting_factor_key: str, check_arrayrecord: bool
) -> bool:
    """Check that replies contain one ArrayRecord, one MetricRecord and that the
    weighting factor key is present.

    These checks assist in keeping the behaviour of Message-based strategies consistent
    with *Ins/*Res-based strategies.
    """
    # Checking for ArrayRecord consistency
    skip_aggregation = False
    if check_arrayrecord:
        if all(len(msg.array_records) != 1 for msg in replies):
            log(
                ERROR,
                "Expected exactly one ArrayRecord in replies, but found more. "
                "Skipping aggregation.",
            )
            skip_aggregation = True
        else:
            # Ensure all key are present in all ArrayRecords
            all_key_sets = [
                set(next(iter(d.array_records.values())).keys()) for d in replies
            ]
            if not all(s == all_key_sets[0] for s in all_key_sets):
                log(
                    ERROR,
                    "All ArrayRecords must have the same keys for aggregation. "
                    "This condition wasn't met. Skipping aggregation.",
                )
                skip_aggregation = True

    # Checking for MetricRecord consistency
    if all(len(msg.metric_records) != 1 for msg in replies):
        log(
            ERROR,
            "Expected exactly one MetricRecord in replies, but found more. "
            "Skipping aggregation.",
        )
        skip_aggregation = True
    else:
        # Ensure all key are present in all MetricRecords
        all_key_sets = [
            set(next(iter(d.metric_records.values())).keys()) for d in replies
        ]
        if not all(s == all_key_sets[0] for s in all_key_sets):
            log(
                ERROR,
                "All MetricRecords must have the same keys for aggregation. "
                "This condition wasn't met. Skipping aggregation.",
            )
            skip_aggregation = True

        # Check one of the sets for the key to perform weighting averaging
        if weighting_factor_key not in all_key_sets[0]:
            log(
                ERROR,
                "The MetricRecord in the reply messages were expecting key "
                f"`{weighting_factor_key}` to perform averaging of "
                "ArrayRecords and MetricRecords. Skipping aggregation.",
            )
            skip_aggregation = True
        else:
            # Check that it is not a list
            if any(
                isinstance(
                    next(iter(d.metric_records.values()))[weighting_factor_key], list
                )
                for d in replies
            ):
                log(
                    ERROR,
                    "The MetricRecord in the reply messages were expecting key "
                    f"`{weighting_factor_key}` to be a single value (float or int), "
                    "but found a list. Skipping aggregation.",
                )
                skip_aggregation = True

    return skip_aggregation
