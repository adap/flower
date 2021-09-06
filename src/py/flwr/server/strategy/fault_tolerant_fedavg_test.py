# Copyright 2020 Adap GmbH. All Rights Reserved.
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


from typing import List, Optional, Tuple
from unittest.mock import MagicMock

from flwr.common import EvaluateRes, FitRes, Parameters, Weights, parameters_to_weights
from flwr.server.client_proxy import ClientProxy

from .fault_tolerant_fedavg import FaultTolerantFedAvg


def test_aggregate_fit_no_results_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.1)
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[BaseException] = []
    expected: Optional[Parameters] = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_no_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.1)
    results: List[Tuple[ClientProxy, FitRes]] = []
    failures: List[BaseException] = [Exception()]
    expected: Optional[Parameters] = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_not_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.5)
    results: List[Tuple[ClientProxy, FitRes]] = [
        (MagicMock(), FitRes(Parameters(tensors=[], tensor_type=""), 1, 1, 0.1))
    ]
    failures: List[BaseException] = [Exception(), Exception()]
    expected: Optional[Parameters] = None

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_fit_just_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.5)
    results: List[Tuple[ClientProxy, FitRes]] = [
        (MagicMock(), FitRes(Parameters(tensors=[], tensor_type=""), 1, 1, 0.1))
    ]
    failures: List[BaseException] = [Exception()]
    expected: Optional[Weights] = []

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    assert parameters_to_weights(actual) == expected


def test_aggregate_fit_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_fit=0.99)
    results: List[Tuple[ClientProxy, FitRes]] = [
        (MagicMock(), FitRes(Parameters(tensors=[], tensor_type=""), 1, 1, 0.1))
    ]
    failures: List[BaseException] = []
    expected: Optional[Weights] = []

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    assert parameters_to_weights(actual) == expected


def test_aggregate_evaluate_no_results_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.1)
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = []
    expected: Optional[float] = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_no_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.1)
    results: List[Tuple[ClientProxy, EvaluateRes]] = []
    failures: List[BaseException] = [Exception()]
    expected: Optional[float] = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_not_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.5)
    results: List[Tuple[ClientProxy, EvaluateRes]] = [
        (MagicMock(), EvaluateRes(loss=2.3, num_examples=1))
    ]
    failures: List[BaseException] = [Exception(), Exception()]
    expected: Optional[float] = None

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_just_enough_results() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.5)
    results: List[Tuple[ClientProxy, EvaluateRes]] = [
        (MagicMock(), EvaluateRes(loss=2.3, num_examples=1))
    ]
    failures: List[BaseException] = [Exception()]
    expected: Optional[float] = 2.3

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected


def test_aggregate_evaluate_no_failures() -> None:
    """Test evaluate function."""
    # Prepare
    strategy = FaultTolerantFedAvg(min_completion_rate_evaluate=0.99)
    results: List[Tuple[ClientProxy, EvaluateRes]] = [
        (MagicMock(), EvaluateRes(loss=2.3, num_examples=1))
    ]
    failures: List[BaseException] = []
    expected: Optional[float] = 2.3

    # Execute
    actual, _ = strategy.aggregate_evaluate(1, results, failures)

    # Assert
    assert actual == expected
