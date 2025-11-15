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
"""FaultTolerantFedAvg tests."""


from unittest.mock import MagicMock

from flwr.common import (
    Code,
    EvaluateRes,
    FitRes,
    NDArrays,
    Parameters,
    Status,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from .fault_tolerant_fedavg import FaultTolerantFedAvg


def test_aggregate_fit_no_results_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.1)
    results: list[tuple[ClientProxy, FitRes]] = []
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = []
    expected: Parameters | None = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_no_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.1)
    results: list[tuple[ClientProxy, FitRes]] = []
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = [Exception()]
    expected: Parameters | None = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_not_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.5)
    results: list[tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                Status(code=Code.OK, message="Success"),
                Parameters(tensors=[], tensor_type=""),
                1,
                {},
            ),
        )
    ]
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = [
        Exception(),
        Exception(),
    ]
    expected: Parameters | None = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_just_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.5)
    results: list[tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                Status(code=Code.OK, message="Success"),
                Parameters(tensors=[], tensor_type=""),
                1,
                {},
            ),
        )
    ]
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = [Exception()]
    expected: NDArrays | None = []

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    assert parameters_to_ndarrays(actual) == expected


def test_aggregate_fit_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.99)
    results: list[tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                Status(code=Code.OK, message="Success"),
                Parameters(tensors=[], tensor_type=""),
                1,
                {},
            ),
        )
    ]
    failures: list[tuple[ClientProxy, FitRes] | BaseException] = []
    expected: NDArrays | None = []

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    assert parameters_to_ndarrays(actual) == expected


def test_aggregate_evaluate_no_results_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.1)
    results: list[tuple[ClientProxy, EvaluateRes]] = []
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = []
    expected: float | None = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_no_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.1)
    results: list[tuple[ClientProxy, EvaluateRes]] = []
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = [Exception()]
    expected: float | None = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_not_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.5)
    results: list[tuple[ClientProxy, EvaluateRes]] = [
        (
            MagicMock(),
            EvaluateRes(
                status=Status(code=Code.OK, message="Success"),
                loss=2.3,
                num_examples=1,
                metrics={},
            ),
        )
    ]
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = [
        Exception(),
        Exception(),
    ]
    expected: float | None = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_just_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.5)
    results: list[tuple[ClientProxy, EvaluateRes]] = [
        (
            MagicMock(),
            EvaluateRes(
                Status(code=Code.OK, message="Success"),
                loss=2.3,
                num_examples=1,
                metrics={},
            ),
        )
    ]
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = [Exception()]
    expected: float | None = 2.3

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.99)
    results: list[tuple[ClientProxy, EvaluateRes]] = [
        (
            MagicMock(),
            EvaluateRes(
                Status(code=Code.OK, message="Success"),
                loss=2.3,
                num_examples=1,
                metrics={},
            ),
        )
    ]
    failures: list[tuple[ClientProxy, EvaluateRes] | BaseException] = []
    expected: float | None = 2.3

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected
