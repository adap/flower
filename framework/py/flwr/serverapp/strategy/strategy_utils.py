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
from logging import INFO
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

from ..exception import InconsistentMessageReplies


def config_to_str(config: ConfigRecord) -> str:
    """Convert a ConfigRecord to a string representation masking bytes."""
    content = ", ".join(
        f"'{k}': {'<bytes>' if isinstance(v, bytes) else v}" for k, v in config.items()
    )
    return f"{{{content}}}"


def log_strategy_start_info(
    num_rounds: int,
    arrays: ArrayRecord,
    train_config: Optional[ConfigRecord],
    evaluate_config: Optional[ConfigRecord],
) -> None:
    """Log information about the strategy start."""
    log(INFO, "\t├── Number of rounds: %d", num_rounds)
    log(
        INFO,
        "\t├── ArrayRecord (%.2f MB)",
        sum(len(array.data) for array in arrays.values()) / (1024**2),
    )
    log(
        INFO,
        "\t├── ConfigRecord (train): %s",
        config_to_str(train_config) if train_config else "(empty!)",
    )
    log(
        INFO,
        "\t├── ConfigRecord (evaluate): %s",
        config_to_str(evaluate_config) if evaluate_config else "(empty!)",
    )


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
        OrderedDict({k: Array(np.asarray(v)) for k, v in aggregated_np_arrays.items()})
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
                        aggregated_metrics[key] = [v * weight for v in value]
                    else:
                        aggregated_metrics[key] = value * weight
                else:
                    if isinstance(value, list):
                        current_list = cast(list[float], aggregated_metrics[key])
                        aggregated_metrics[key] = [
                            curr + val * weight
                            for curr, val in zip(current_list, value)
                        ]
                    else:
                        current_value = cast(float, aggregated_metrics[key])
                        aggregated_metrics[key] = current_value + value * weight

    return aggregated_metrics


def sample_nodes(
    grid: Grid, min_available_nodes: int, sample_size: int
) -> tuple[list[int], list[int]]:
    """Sample the specified number of nodes using the Grid.

    Parameters
    ----------
    grid : Grid
        The grid object.
    min_available_nodes : int
        The minimum number of available nodes to sample from.
    sample_size : int
        The number of nodes to sample.

    Returns
    -------
    tuple[list[int], list[int]]
        A tuple containing the sampled node IDs and the list
        of all connected node IDs.
    """
    sampled_nodes = []

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

    # Sample nodes
    sampled_nodes = random.sample(all_nodes, sample_size)

    return sampled_nodes, all_nodes


# pylint: disable=too-many-return-statements
def validate_message_reply_consistency(
    replies: list[RecordDict], weighted_by_key: str, check_arrayrecord: bool
) -> None:
    """Validate that replies contain exactly one ArrayRecord and one MetricRecord, and
    that the MetricRecord includes a weight factor key.

    These checks ensure that Message-based strategies behave consistently with
    *Ins/*Res-based strategies.
    """
    # Checking for ArrayRecord consistency
    if check_arrayrecord:
        if any(len(msg.array_records) != 1 for msg in replies):
            raise InconsistentMessageReplies(
                reason="Expected exactly one ArrayRecord in replies. "
                "Skipping aggregation."
            )

        # Ensure all key are present in all ArrayRecords
        record_key = next(iter(replies[0].array_records.keys()))
        all_keys = set(replies[0][record_key].keys())
        if any(set(msg.get(record_key, {}).keys()) != all_keys for msg in replies[1:]):
            raise InconsistentMessageReplies(
                reason="All ArrayRecords must have the same keys for aggregation. "
                "This condition wasn't met. Skipping aggregation."
            )

    # Checking for MetricRecord consistency
    if any(len(msg.metric_records) != 1 for msg in replies):
        raise InconsistentMessageReplies(
            reason="Expected exactly one MetricRecord in replies, but found more. "
            "Skipping aggregation."
        )

    # Ensure all key are present in all MetricRecords
    record_key = next(iter(replies[0].metric_records.keys()))
    all_keys = set(replies[0][record_key].keys())
    if any(set(msg.get(record_key, {}).keys()) != all_keys for msg in replies[1:]):
        raise InconsistentMessageReplies(
            reason="All MetricRecords must have the same keys for aggregation. "
            "This condition wasn't met. Skipping aggregation."
        )

    # Verify the weight factor key presence in all MetricRecords
    if weighted_by_key not in all_keys:
        raise InconsistentMessageReplies(
            reason=f"Missing required key `{weighted_by_key}` in the MetricRecord of "
            "reply messages. Cannot average ArrayRecords and MetricRecords. Skipping "
            "aggregation."
        )

    # Check that it is not a list
    if any(isinstance(msg[record_key][weighted_by_key], list) for msg in replies):
        raise InconsistentMessageReplies(
            reason=f"Key `{weighted_by_key}` in the MetricRecord of reply messages "
            "must be a single value (int or float), but a list was found. Skipping "
            "aggregation."
        )
