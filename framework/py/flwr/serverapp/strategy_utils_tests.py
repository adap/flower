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
"""Tests for message-based strategy utilities."""


import numpy as np

from flwr.common import ArrayRecord, ConfigRecord, MetricRecord, RecordDict

from .strategy_utils import aggregate_arrayrecords, config_to_str


def test_config_to_str() -> None:
    """Test that items of types bytes are masked out."""
    config = ConfigRecord({"a": 123, "b": [1, 2, 3], "c": b"bytes"})
    expected_str = "{'a': 123, 'b': [1, 2, 3], 'c': <bytes>}"
    assert config_to_str(config) == expected_str


def test_arrayrecords_aggregation() -> None:
    """Test aggregation of ArrayRecords."""
    num_replies = 3
    num_arrays = 4
    weights = [0.25, 0.4, 0.35]
    np_arrays = [
        [np.random.randn(7, 3) for _ in range(num_arrays)] for _ in range(num_replies)
    ]

    avg_list = [
        np.average([lst[i] for lst in np_arrays], axis=0, weights=weights)
        for i in range(num_arrays)
    ]

    # Construct RecordDicts (mimicing replies)
    records = [
        RecordDict(
            {
                "arrays": ArrayRecord(np_arrays[i]),
                "metrics": MetricRecord({"weight": weights[i]}),
            }
        )
        for i in range(num_replies)
    ]
    # Execute aggregate
    aggrd = aggregate_arrayrecords(records, weighting_metric_name="weight")

    # Assert consistency
    assert all(np.allclose(a, b) for a, b in zip(aggrd.to_numpy_ndarrays(), avg_list))
    assert aggrd.object_id == ArrayRecord(avg_list).object_id
