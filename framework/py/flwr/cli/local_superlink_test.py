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

_IS_STARTED_PATH = "flwr.cli.local_superlink._is_local_superlink_started"
_START_PATH = "flwr.cli.local_superlink._start_local_superlink"


def test_options_only_connection_uses_local_superlink() -> None:
    """Options-only connections are mapped to local SuperLink."""
    connection = SuperLinkConnection(
        name="local",
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(_IS_STARTED_PATH, return_value=True) as mock_is_started:
        with patch(_START_PATH) as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == "127.0.0.1:39093"
    assert resolved.insecure is True
    assert resolved.root_certificates is None
    mock_is_started.assert_called_once()
    mock_start.assert_not_called()


def test_options_only_connection_starts_local_superlink_when_unavailable() -> None:
    """Local SuperLink is started when it is unavailable."""
    connection = SuperLinkConnection(
        name="local",
        options=SuperLinkSimulationOptions(num_supernodes=2),
    )

    with patch(_IS_STARTED_PATH, return_value=False):
        with patch(_START_PATH) as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved.address == "127.0.0.1:39093"
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

    with patch(_IS_STARTED_PATH) as mock_is_started:
        with patch(_START_PATH) as mock_start:
            resolved = ensure_local_superlink(connection)

    assert resolved is connection
    mock_is_started.assert_not_called()
    mock_start.assert_not_called()
