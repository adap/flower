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
"""FedRDF tests."""


from unittest.mock import MagicMock

import numpy as np
from numpy import array, float32

from flwr.common import (
    Code,
    FitRes,
    Parameters,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

from .fedrdf import FedRDF


def test_fedrdf_num_fit_clients_20_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = FedRDF()
    expected = 20

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=20)

    # Assert
    assert expected == actual


def test_fedrdf_num_fit_clients_19_available() -> None:
    """Test num_fit_clients function with 19 available clients."""
    # Prepare
    strategy = FedRDF()
    expected = 19

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=19)

    # Assert
    assert expected == actual


def test_fedrdf_aggregate_fit_with_threshold_zero() -> None:
    """Test aggregate_fit with threshold=0 (always use FFT)."""
    # Prepare
    strategy = FedRDF(threshold=0.0)

    # Create mock clients
    clients = [MagicMock(spec=ClientProxy) for _ in range(3)]

    # Create simple parameters
    params1 = ndarrays_to_parameters([array([[1.0, 2.0]], dtype=float32)])
    params2 = ndarrays_to_parameters([array([[2.0, 3.0]], dtype=float32)])
    params3 = ndarrays_to_parameters([array([[3.0, 4.0]], dtype=float32)])

    results = [
        (clients[0], FitRes(Status(Code.OK, ""), params1, 10, {})),
        (clients[1], FitRes(Status(Code.OK, ""), params2, 10, {})),
        (clients[2], FitRes(Status(Code.OK, ""), params3, 10, {})),
    ]

    # Execute
    params_aggregated, metrics = strategy.aggregate_fit(
        server_round=1, results=results, failures=[]
    )

    # Assert
    assert params_aggregated is not None
    assert isinstance(metrics, dict)

    # Verify aggregated parameters have correct shape
    aggregated_arrays = parameters_to_ndarrays(params_aggregated)
    assert len(aggregated_arrays) == 1
    assert aggregated_arrays[0].shape == (1, 2)


def test_fedrdf_aggregate_fit_no_results() -> None:
    """Test aggregate_fit with no results."""
    # Prepare
    strategy = FedRDF()

    # Execute
    params_aggregated, metrics = strategy.aggregate_fit(
        server_round=1, results=[], failures=[]
    )

    # Assert
    assert params_aggregated is None
    assert metrics == {}


def test_fedrdf_aggregate_fit_with_failures() -> None:
    """Test aggregate_fit with failures and accept_failures=False."""
    # Prepare
    strategy = FedRDF(accept_failures=False)

    # Create mock client and parameters
    client = MagicMock(spec=ClientProxy)
    params = ndarrays_to_parameters([array([[1.0, 2.0]], dtype=float32)])

    results = [
        (client, FitRes(Status(Code.OK, ""), params, 10, {})),
    ]
    failures = [BaseException("Test failure")]

    # Execute
    params_aggregated, metrics = strategy.aggregate_fit(
        server_round=1, results=results, failures=failures
    )

    # Assert - should return None when failures not accepted
    assert params_aggregated is None
    assert metrics == {}


def test_fedrdf_ks_proportion() -> None:
    """Test ks_proportion method."""
    # Prepare
    strategy = FedRDF()

    # Create a sample with normal distribution
    np.random.seed(42)
    sample = np.random.normal(0, 1, 100)

    # Execute
    proportion = strategy.ks_proportion(sample)

    # Assert - proportion should be between 0 and 1
    assert 0.0 <= proportion <= 1.0


def test_fedrdf_skewness() -> None:
    """Test skewness method."""
    # Prepare
    strategy = FedRDF()

    # Create arrays with similar distributions (low skewness)
    np.random.seed(42)
    arrays = [
        np.random.normal(0, 1, (5, 5)) for _ in range(3)
    ]

    # Execute
    skewness = strategy.skewness(arrays)

    # Assert - skewness should be between 0 and 1
    assert 0.0 <= skewness <= 1.0


def test_fedrdf_fourier_aggregate() -> None:
    """Test fourier_aggregate method."""
    # Prepare
    strategy = FedRDF()

    # Create simple arrays
    arrays = [
        array([[1.0, 2.0], [3.0, 4.0]], dtype=float32),
        array([[2.0, 3.0], [4.0, 5.0]], dtype=float32),
        array([[3.0, 4.0], [5.0, 6.0]], dtype=float32),
    ]

    # Execute
    result = strategy.fourier_aggregate(arrays)

    # Assert
    assert result.shape == arrays[0].shape
    assert result.dtype == arrays[0].dtype


def test_fedrdf_aggregate_fedrdf_with_high_threshold() -> None:
    """Test aggregate_fedrdf with high threshold (should use FedAvg)."""
    # Prepare
    strategy = FedRDF(threshold=1.0)  # High threshold

    # Create simple arrays with low skewness
    arrays1 = [array([[1.0, 2.0]], dtype=float32)]
    arrays2 = [array([[2.0, 3.0]], dtype=float32)]
    arrays3 = [array([[3.0, 4.0]], dtype=float32)]

    results = [
        (arrays1, 10),
        (arrays2, 10),
        (arrays3, 10),
    ]

    # Execute
    aggregated = strategy.aggregate_fedrdf(results)

    # Assert
    assert len(aggregated) == 1
    assert aggregated[0].shape == (1, 2)

    # With equal weights, should be close to average
    expected_avg = array([[2.0, 3.0]], dtype=float32)
    np.testing.assert_array_almost_equal(aggregated[0], expected_avg, decimal=5)


def test_fedrdf_repr() -> None:
    """Test string representation."""
    # Prepare
    strategy = FedRDF(threshold=0.5, accept_failures=False)

    # Execute
    repr_str = repr(strategy)

    # Assert
    assert "FedRDF" in repr_str
    assert "accept_failures=False" in repr_str
    assert "threshold=0.5" in repr_str
