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
"""Flower Client app tests."""


from typing import Dict, Tuple

from flwr.common import (
    Config,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    GetPropertiesIns,
    GetPropertiesRes,
    NDArrays,
    Scalar,
)

from .app import ClientLike, start_client, start_numpy_client, to_client
from .client import Client
from .numpy_client import NumPyClient


class PlainClient(Client):
    """Client implementation extending the low-level Client."""

    def get_properties(self, ins: GetPropertiesIns) -> GetPropertiesRes:
        """Raise an Exception because this method is not expected to be called."""
        raise Exception()

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Raise an Exception because this method is not expected to be called."""
        raise Exception()

    def fit(self, ins: FitIns) -> FitRes:
        """Raise an Exception because this method is not expected to be called."""
        raise Exception()

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Raise an Exception because this method is not expected to be called."""
        raise Exception()


class NeedsWrappingClient(NumPyClient):
    """Client implementation extending the high-level NumPyClient."""

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Raise an Exception because this method is not expected to be called."""
        raise Exception()

    def get_parameters(self, config: Config) -> NDArrays:
        """Raise an Exception because this method is not expected to be called."""
        raise Exception()

    def fit(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Raise an Exception because this method is not expected to be called."""
        raise Exception()

    def evaluate(
        self, parameters: NDArrays, config: Config
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Raise an Exception because this method is not expected to be called."""
        raise Exception()


def test_to_client_with_client() -> None:
    """Test to_client."""
    # Prepare
    client_like: ClientLike = PlainClient()

    # Execute
    actual = to_client(client_like=client_like)

    # Assert
    assert isinstance(actual, Client)


def test_to_client_with_numpyclient() -> None:
    """Test fit_clients."""
    # Prepare
    client_like: ClientLike = NeedsWrappingClient()

    # Execute
    actual = to_client(client_like=client_like)

    # Assert
    assert isinstance(actual, Client)


def test_start_client_transport_invalid() -> None:
    """Test start_client(..., transport=...)."""
    # Prepare
    client: Client = PlainClient()
    invalid_transport = "invalid-transport-value"

    # Execute
    try:
        start_client(
            server_address="0.0.0.0:8080", client=client, transport=invalid_transport
        )
        raise AssertionError()  # Fail the test if no exception was raised
    except ValueError:
        pass


def test_start_numpy_client_transport_invalid() -> None:
    """Test start_client(..., transport=...)."""
    # Prepare
    client: NumPyClient = NeedsWrappingClient()
    invalid_transport = "invalid-transport-value"

    # Execute
    try:
        start_numpy_client(
            server_address="0.0.0.0:8080", client=client, transport=invalid_transport
        )
        raise AssertionError()  # Fail the test if no exception was raised
    except ValueError:
        pass
