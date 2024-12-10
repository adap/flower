# Copyright 2022 Flower Labs GmbH. All Rights Reserved.
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
"""Flower NumPyClient tests."""

import numpy as np

from flwr.common import Config, NDArrays, Properties, Scalar

from .numpy_client import (
    NumPyClient,
    has_evaluate,
    has_fit,
    has_get_parameters,
    has_get_properties,
)


class OverridingClient(NumPyClient):
    """Client overriding `get_properties`."""

    def get_properties(self, config: Config) -> Properties:
        """Get empty properties of the client."""
        return {}

    def get_parameters(self, config: Config) -> NDArrays:
        """Get empty parameters of the client."""
        return []

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Simulate training by returning empty weights, 0 samples, empty metrics."""
        return [], 0, {}

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Simulate evaluate by returning 0.0 loss, 0 samples, empty metrics."""
        return 0.0, 0, {}


class NotOverridingClient(NumPyClient):
    """Client not overriding any NumPyClient method."""


def test_has_get_properties_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_get_properties(client=client)

    # Assert
    assert actual == expected


def test_has_get_properties_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_get_properties(client=client)

    # Assert
    assert actual == expected


def test_has_get_parameters_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_get_parameters(client=client)

    # Assert
    assert actual == expected


def test_has_get_parameters_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_get_parameters(client=client)

    # Assert
    assert actual == expected


def test_has_fit_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_fit(client=client)

    # Assert
    assert actual == expected


def test_has_fit_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_fit(client=client)

    # Assert
    assert actual == expected


def test_has_evaluate_true() -> None:
    """Test fit_clients."""
    # Prepare
    client = OverridingClient()
    expected = True

    # Execute
    actual = has_evaluate(client=client)

    # Assert
    assert actual == expected


def test_has_evaluate_false() -> None:
    """Test fit_clients."""
    # Prepare
    client = NotOverridingClient()
    expected = False

    # Execute
    actual = has_evaluate(client=client)

    # Assert
    assert actual == expected


def test_fit_return_type() -> None:
    """Test that fit returns the correct type."""
    # Prepare
    client = OverridingClient()

    # Execute
    parameters, num_examples, metrics = client.fit(
        parameters=[np.array([0.1, 0.2])], config={"epochs": 5}
    )

    # Assert
    # Check if parameters is a list and all elements are np.ndarray
    assert isinstance(parameters, list)
    assert all(isinstance(p, np.ndarray) for p in parameters)

    # Check other return types
    assert isinstance(num_examples, int)
    assert isinstance(metrics, dict)
    assert all(
        isinstance(k, str) and isinstance(v, (bool, bytes, float, int, str))
        for k, v in metrics.items()
    )


def test_evaluate_return_type() -> None:
    """Test that evaluate returns the correct type."""
    # Prepare
    client = OverridingClient()

    # Execute
    loss, num_examples, metrics = client.evaluate(
        parameters=[np.array([0.1, 0.2])], config={"batch_size": 32}
    )

    # Assert
    assert isinstance(loss, float)
    assert isinstance(num_examples, int)
    assert isinstance(metrics, dict)
    assert all(isinstance(k, str) for k in metrics)  # Fix: Removed `.keys()`
    assert all(isinstance(v, (bool, bytes, float, int, str)) for v in metrics.values())


def test_get_parameters_return_type() -> None:
    """Test that get_parameters returns the correct type."""
    # Prepare
    client = OverridingClient()

    # Execute
    parameters = client.get_parameters(config={})

    # Assert
    # Check if parameters is a list and all elements are np.ndarray
    assert isinstance(parameters, list)
    assert all(isinstance(p, np.ndarray) for p in parameters)


def test_get_properties_return_type() -> None:
    """Test that get_properties returns the correct type."""
    # Prepare
    client = OverridingClient()

    # Execute
    properties = client.get_properties(config={})

    # Assert
    assert isinstance(properties, dict)  # Properties is a dict[str, Scalar]
    assert all(isinstance(k, str) for k in properties)
    assert all(
        isinstance(v, (bool, bytes, float, int, str)) for v in properties.values()
    )
