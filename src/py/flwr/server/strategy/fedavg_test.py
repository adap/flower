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
"""FedAvg tests."""

from typing import List, Tuple
from unittest.mock import MagicMock

from numpy import array, float32
from numpy.testing import assert_almost_equal

from flwr.common import FitRes, Weights, parameters_to_weights
from flwr.common.parameter import weights_to_parameters
from flwr.server.client_proxy import ClientProxy

from .fedavg import FedAvg


def test_fedavg_num_fit_clients_20_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedAvg()
    expected = 2

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=20)

    # Assert
    assert expected == actual


def test_fedavg_num_fit_clients_19_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedAvg()
    expected = 2

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=19)

    # Assert
    assert expected == actual


def test_fedavg_num_fit_clients_10_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedAvg()
    expected = 2

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=10)

    # Assert
    assert expected == actual


def test_fedavg_num_fit_clients_minimum() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedAvg()
    expected = 2

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=9)

    # Assert
    assert expected == actual


def test_fedavg_num_evaluation_clients_40_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = FedAvg(fraction_eval=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=40)

    # Assert
    assert expected == actual


def test_fedavg_num_evaluation_clients_39_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = FedAvg(fraction_eval=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=39)

    # Assert
    assert expected == actual


def test_fedavg_num_evaluation_clients_20_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = FedAvg(fraction_eval=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=20)

    # Assert
    assert expected == actual


def test_fedavg_num_evaluation_clients_minimum() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = FedAvg(fraction_eval=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=19)

    # Assert
    assert expected == actual


def test_aggregate_fit_using_near_one_server_lr_and_no_momentum() -> None:
    """Test aggregate with near-one learning rate and no momentum."""
    # Prepare
    weights0_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights0_1 = array([7, 8, 9, 10], dtype=float32)
    weights1_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights1_1 = array([7, 8, 9, 10], dtype=float32)

    initial_weights: Weights = [
        array([[0, 0, 0], [0, 0, 0]], dtype=float32),
        array([0, 0, 0, 0], dtype=float32),
    ]

    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(weights_to_parameters([weights0_0, weights0_1]), 1, {}),
        ),
        (
            MagicMock(),
            FitRes(weights_to_parameters([weights1_0, weights1_1]), 2, {}),
        ),
    ]
    failures: List[BaseException] = []
    expected: Weights = [
        array([[1, 2, 3], [4, 5, 6]], dtype=float32),
        array([7, 8, 9, 10], dtype=float32),
    ]

    strategy = FedAvg(
        initial_parameters=weights_to_parameters(initial_weights),
        server_learning_rate=1.0 + 1e-9,
    )

    # Execute
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Assert
    assert actual
    for w_act, w_exp in zip(parameters_to_weights(actual), expected):
        assert_almost_equal(w_act, w_exp)


def test_aggregate_fit_server_learning_rate_and_momentum() -> None:
    """Test aggregate with near-one learning rate and near-zero momentum."""
    # Prepare
    weights0_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights0_1 = array([7, 8, 9, 10], dtype=float32)
    weights1_0 = array([[1, 2, 3], [4, 5, 6]], dtype=float32)
    weights1_1 = array([7, 8, 9, 10], dtype=float32)

    initial_weights: Weights = [
        array([[0, 0, 0], [0, 0, 0]], dtype=float32),
        array([0, 0, 0, 0], dtype=float32),
    ]

    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(weights_to_parameters([weights0_0, weights0_1]), 1, {}),
        ),
        (
            MagicMock(),
            FitRes(weights_to_parameters([weights1_0, weights1_1]), 2, {}),
        ),
    ]
    failures: List[BaseException] = []
    expected: Weights = [
        array([[1, 2, 3], [4, 5, 6]], dtype=float32),
        array([7, 8, 9, 10], dtype=float32),
    ]

    strategy = FedAvg(
        initial_parameters=weights_to_parameters(initial_weights),
        server_learning_rate=1.0 + 1e-9,
        server_momentum=1.0e-9,
    )

    # Execute
    # First round (activate momentum)
    actual, _ = strategy.aggregate_fit(1, results, failures)

    # Second round (update momentum)
    actual, _ = strategy.aggregate_fit(2, results, failures)

    # Assert
    assert actual
    for w_act, w_exp in zip(parameters_to_weights(actual), expected):
        assert_almost_equal(w_act, w_exp)
