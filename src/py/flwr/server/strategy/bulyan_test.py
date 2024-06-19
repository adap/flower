# Copyright 2023 Flower Labs GmbH. All Rights Reserved.
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
"""Bulyan tests."""


from typing import List, Tuple
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
from flwr.server.client_proxy import ClientProxy

from .bulyan import Bulyan


# pylint: disable=too-many-locals
def test_aggregate_fit() -> None:
    """Tests if Bulyan is aggregating correctly."""
    # Prepare
    previous_weights: NDArrays = [array([0.1, 0.1, 0.1, 0.1], dtype=float32)]
    strategy = Bulyan(
        initial_parameters=ndarrays_to_parameters(previous_weights),
        num_malicious_clients=0,
        to_keep=0,
    )
    param_0: Parameters = ndarrays_to_parameters(
        [array([0.2, 0.2, 0.2, 0.2], dtype=float32)]
    )
    param_1: Parameters = ndarrays_to_parameters(
        [array([0.5, 0.5, 0.5, 0.5], dtype=float32)]
    )
    param_2: Parameters = ndarrays_to_parameters(
        [array([0.7, 0.7, 0.7, 0.7], dtype=float32)]
    )
    param_3: Parameters = ndarrays_to_parameters(
        [array([12.0, 12.0, 12.0, 12.0], dtype=float32)]
    )
    param_4: Parameters = ndarrays_to_parameters(
        [array([0.1, 0.1, 0.1, 0.1], dtype=float32)]
    )
    param_5: Parameters = ndarrays_to_parameters(
        [array([0.1, 0.1, 0.1, 0.1], dtype=float32)]
    )
    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_0,
                num_examples=5,
                metrics={},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_1,
                num_examples=5,
                metrics={},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_2,
                num_examples=5,
                metrics={},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_3,
                num_examples=5,
                metrics={},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_4,
                num_examples=5,
                metrics={},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=param_5,
                num_examples=5,
                metrics={},
            ),
        ),
    ]
    coordinate = (0.2 + 0.5 + 0.7 + 12.0 + 0.1 + 0.1) / 6
    expected: NDArrays = [array([coordinate] * 4, dtype=float32)]

    # Execute
    actual_aggregated, _ = strategy.aggregate_fit(
        server_round=1, results=results, failures=[]
    )
    if actual_aggregated:
        actual_list = parameters_to_ndarrays(actual_aggregated)
        actual = actual_list[0]
    assert (actual == expected[0]).all()