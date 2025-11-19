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
"""Tests for NoOpFederationManager."""


from unittest.mock import Mock

import pytest

from flwr.common.constant import NOOP_FLWR_AID
from flwr.common.typing import Federation, Run, RunStatus
from flwr.proto.node_pb2 import NodeInfo  # pylint: disable=E0611
from flwr.supercore.constant import NOOP_FEDERATION

from .noop_federation_manager import NoOpFederationManager


def test_get_details_with_valid_federation() -> None:
    """Test get_details returns correct Federation details."""
    # Prepare
    manager = NoOpFederationManager()
    mock_linkstate = Mock()
    manager.linkstate = mock_linkstate

    # Mock data
    run_id_1 = 123
    run_id_2 = 456
    mock_run_1 = Run(
        run_id=run_id_1,
        fab_id="test_fab_1",
        fab_version="1.0.0",
        fab_hash="hash123",
        override_config={},
        pending_at="2025-01-01T00:00:00",
        starting_at="2025-01-01T00:01:00",
        running_at="2025-01-01T00:02:00",
        finished_at="",
        status=RunStatus(status="running", sub_status="", details=""),
        flwr_aid=NOOP_FLWR_AID,
        federation=NOOP_FEDERATION,
    )
    mock_run_2 = Run(
        run_id=run_id_2,
        fab_id="test_fab_2",
        fab_version="2.0.0",
        fab_hash="hash456",
        override_config={},
        pending_at="2025-01-02T00:00:00",
        starting_at="2025-01-02T00:01:00",
        running_at="2025-01-02T00:02:00",
        finished_at="2025-01-02T00:10:00",
        status=RunStatus(status="finished", sub_status="", details=""),
        flwr_aid=NOOP_FLWR_AID,
        federation=NOOP_FEDERATION,
    )
    mock_node_1 = NodeInfo(
        node_id=1,
        owner_aid=NOOP_FLWR_AID,
        owner_name="test_owner_1",
        status="registered",
        registered_at="2025-01-01T00:00:00",
        heartbeat_interval=30.0,
        public_key=b"public_key_1",
    )
    mock_node_2 = NodeInfo(
        node_id=2,
        owner_aid=NOOP_FLWR_AID,
        owner_name="test_owner_2",
        status="registered",
        registered_at="2025-01-02T00:00:00",
        heartbeat_interval=30.0,
        public_key=b"public_key_2",
    )

    # Configure mocks
    mock_linkstate.get_run_ids.return_value = {run_id_1, run_id_2}
    mock_linkstate.get_node_info.return_value = [mock_node_1, mock_node_2]
    mock_linkstate.get_run.side_effect = lambda run_id: (
        mock_run_1 if run_id == run_id_1 else mock_run_2
    )

    # Execute
    result = manager.get_details(NOOP_FEDERATION)

    # Assert
    assert isinstance(result, Federation)
    assert result.name == NOOP_FEDERATION
    assert result.member_aids == [NOOP_FLWR_AID]
    assert len(result.nodes) == 2
    assert mock_node_1 in result.nodes and mock_node_2 in result.nodes
    assert len(result.runs) == 2
    assert mock_run_1 in result.runs and mock_run_2 in result.runs


def test_get_details_with_invalid_federation() -> None:
    """Test get_details raises ValueError for invalid federation."""
    # Prepare
    manager = NoOpFederationManager()
    mock_linkstate = Mock()
    manager.linkstate = mock_linkstate
    invalid_federation = "invalid_federation"

    # Execute & Assert
    with pytest.raises(ValueError):
        manager.get_details(invalid_federation)


def test_get_details_with_no_runs() -> None:
    """Test get_details returns empty runs list when no runs exist."""
    # Prepare
    manager = NoOpFederationManager()
    mock_linkstate = Mock()
    manager.linkstate = mock_linkstate

    # Configure mocks for empty runs
    mock_linkstate.get_run_ids.return_value = set()
    mock_linkstate.get_node_info.return_value = []

    # Execute
    result = manager.get_details(NOOP_FEDERATION)

    # Assert
    assert result.name == NOOP_FEDERATION
    assert result.member_aids == [NOOP_FLWR_AID]
    assert len(result.nodes) == 0
    assert len(result.runs) == 0


def test_exists() -> None:
    """Test exists method returns True only for NOOP_FEDERATION."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.exists(NOOP_FEDERATION) is True
    assert manager.exists("other_federation") is False


def test_has_member() -> None:
    """Test has_member method always returns True."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.has_member("any_aid", NOOP_FEDERATION) is True
    assert manager.has_member("another_aid", "any_federation") is True


def test_filter_nodes() -> None:
    """Test filter_nodes method returns all provided node IDs."""
    # Prepare
    manager = NoOpFederationManager()
    node_ids = {1, 2, 3, 4, 5}

    # Execute
    result = manager.filter_nodes(node_ids, NOOP_FEDERATION)

    # Assert
    assert result == node_ids


def test_has_node() -> None:
    """Test has_node method always returns True."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.has_node(1, NOOP_FEDERATION) is True
    assert manager.has_node(999, "any_federation") is True


def test_get_federations() -> None:
    """Test get_federations method returns NOOP_FEDERATION."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute
    result = manager.get_federations("any_aid")

    # Assert
    assert result == [NOOP_FEDERATION]
