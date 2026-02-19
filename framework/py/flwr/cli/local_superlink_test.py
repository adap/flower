# Copyright 2026 Flower Labs GmbH. All Rights Reserved.
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
"""Tests for local SuperLink runtime helpers."""


from unittest.mock import patch

from flwr.cli.typing import SuperLinkConnection, SuperLinkSimulationOptions

from .local_superlink import ensure_local_superlink


def test_options_only_connection_uses_runtime_defaults() -> None:
    """Options-only connections are mapped to managed local runtime."""
    connection = SuperLinkConnection(
        name="local",
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(
        "flwr.cli.local_superlink._is_control_api_available", return_value=True
    ) as mock_available:
        with patch("flwr.cli.local_superlink._start_local_superlink") as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == "127.0.0.1:9093"
    assert resolved.insecure is True
    assert resolved.root_certificates is None
    mock_available.assert_called_once()
    mock_start.assert_not_called()


def test_options_only_connection_starts_runtime_when_unavailable() -> None:
    """Managed local runtime is started when Control API is unavailable."""
    connection = SuperLinkConnection(
        name="local",
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(
        "flwr.cli.local_superlink._is_control_api_available", return_value=False
    ), patch("flwr.cli.local_superlink._start_local_superlink") as mock_start:
        resolved = ensure_local_superlink(connection)

    assert resolved.address == "127.0.0.1:9093"
    assert resolved.insecure is True
    mock_start.assert_called_once()


def test_local_config_is_preserved_when_endpoint_available() -> None:
    """Keep explicit-address connection unchanged."""
    connection = SuperLinkConnection(
        name="local-dev",
        address="127.0.0.1:9093",
        insecure=False,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch("flwr.cli.local_superlink._is_control_api_available") as mock_available:
        with patch("flwr.cli.local_superlink._start_local_superlink") as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved is connection
    mock_available.assert_not_called()
    mock_start.assert_not_called()


def test_explicit_local_address_does_not_auto_start() -> None:
    """Explicit local addresses are treated as user-managed."""
    connection = SuperLinkConnection(
        name="local-dev",
        address="localhost:9093",
        insecure=False,
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch("flwr.cli.local_superlink._is_control_api_available") as mock_available:
        with patch("flwr.cli.local_superlink._start_local_superlink") as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == "localhost:9093"
    assert resolved.insecure is False
    mock_available.assert_not_called()
    mock_start.assert_not_called()


def test_explicit_bind_all_address_does_not_auto_start() -> None:
    """Explicit bind-all local addresses are treated as user-managed."""
    connection = SuperLinkConnection(
        name="local-dev",
        address="0.0.0.0:9093",
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch("flwr.cli.local_superlink._is_control_api_available") as mock_available:
        with patch("flwr.cli.local_superlink._start_local_superlink") as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == "0.0.0.0:9093"
    assert resolved.insecure is False
    mock_available.assert_not_called()
    mock_start.assert_not_called()


def test_remote_connection_is_untouched() -> None:
    """Remote connections are returned unchanged."""
    connection = SuperLinkConnection(
        name="remote-sim",
        address="example.com:9093",
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch("flwr.cli.local_superlink._is_control_api_available") as mock_probe:
        resolved = ensure_local_superlink(connection)

    assert resolved is connection
    mock_probe.assert_not_called()
