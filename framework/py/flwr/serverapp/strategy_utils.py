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
from logging import INFO
from time import sleep
from typing import cast

import numpy as np

from flwr.common import Array, ArrayRecord, MetricRecord, NDArray, RecordDict, log
from flwr.server import Grid


@dataclass
class StrategyResults:
    """Data class carrying records generated during the execution of a strategy."""

    arrays: ArrayRecord = field(default_factory=ArrayRecord)
    train_metrics: dict[int, MetricRecord] = field(default_factory=dict)
    evaluate_metrics: dict[int, MetricRecord] = field(default_factory=dict)
    central_evaluate_metrics: dict[int, MetricRecord] = field(default_factory=dict)


def aggregate_arrayrecords(
    records: list[RecordDict], weighting_metric_name: str
) -> ArrayRecord:
    """Perform weighted aggregation all ArrayRecords using a specific key."""
    # Retrieve weighting factor from MetricRecord
    weights: list[float] = []
    for record in records:
        # Get the first (and only) MetricRecord in the record
        metricrecord = next(iter(record.metric_records.values()))
        w = cast(int, metricrecord[weighting_metric_name])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    # Perform weighted aggregation
    aggregated_np_arrays: dict[str, NDArray] = {}

    for record, weight in zip(records, weight_factors):
        for record_item in record.values():
            # For ArrayRecord
            if isinstance(record_item, ArrayRecord):
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
        w = cast(int, metricrecord[weighting_metric_name])
        weights.append(w)

    # Average
    total_weight = sum(weights)
    weight_factors = [w / total_weight for w in weights]

    aggregated_metrics = MetricRecord()
    for record, weight in zip(records, weight_factors):
        for record_item in record.values():
            # For MetricRecord
            if isinstance(record_item, MetricRecord):
                # aggregate in-place
                for key, value in record_item.items():
                    if key == weighting_metric_name:
                        # We exclude the weighting key from the aggregated MetricRecord
                        continue
                    if key not in aggregated_metrics:
                        if isinstance(value, list):
                            aggregated_metrics[key] = (
                                np.array(value) * weight
                            ).tolist()
                        else:
                            aggregated_metrics[key] = value * weight
                    else:
                        if isinstance(value, list):
                            aggregated_metrics[key] = (
                                np.array(aggregated_metrics[key])
                                + np.array(value) * weight
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
