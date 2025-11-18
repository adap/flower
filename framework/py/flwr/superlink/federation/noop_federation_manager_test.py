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


import pytest

from flwr.common.constant import NOOP_FLWR_AID
from flwr.supercore.constant import NOOP_FEDERATION

from .noop_federation_manager import NoOpFederationManager


def test_exists() -> None:
    """Test exists method returns True only for NOOP_FEDERATION."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.exists(NOOP_FEDERATION)
    assert not manager.exists("other_federation")


def test_has_member() -> None:
    """Test has_member method returns True only for NOOP_FLWR_AID."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.has_member(NOOP_FLWR_AID, NOOP_FEDERATION) is True
    assert manager.has_member("any_aid", NOOP_FEDERATION) is False

    # Test that it raises ValueError for non-existent federation
    with pytest.raises(ValueError):
        manager.has_member("any_aid", "other_federation")


def test_filter_nodes() -> None:
    """Test filter_nodes method returns all provided node IDs."""
    # Prepare
    manager = NoOpFederationManager()
    node_ids = {1, 2, 3, 4, 5}

    # Execute
    result = manager.filter_nodes(node_ids, NOOP_FEDERATION)

    # Assert
    assert result == node_ids

    # Test that it raises ValueError for non-existent federation
    with pytest.raises(ValueError):
        manager.filter_nodes(node_ids, "other_federation")


def test_has_node() -> None:
    """Test has_node method returns True for NOOP_FEDERATION."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute & Assert
    assert manager.has_node(1, NOOP_FEDERATION) is True
    assert manager.has_node(999, NOOP_FEDERATION) is True

    # Test that it raises ValueError for non-existent federation
    with pytest.raises(ValueError):
        manager.has_node(999, "any_federation")


def test_get_federations() -> None:
    """Test get_federations method returns NOOP_FEDERATION."""
    # Prepare
    manager = NoOpFederationManager()

    # Execute
    result = manager.get_federations("any_aid")
    result2 = manager.get_federations(NOOP_FLWR_AID)

    # Assert
    assert len(result) == 0
    assert result2 == [NOOP_FEDERATION]
