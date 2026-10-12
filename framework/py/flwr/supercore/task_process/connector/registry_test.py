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
"""Tests for the connector registry."""

from unittest.mock import Mock

import pytest

from flwr.supercore.typing import JSONObject

from . import registry
from .definition import (
    ActionAccess,
    ActionDefinition,
    ConnectorDefinition,
    ConnectorExecutionContext,
    ConnectorToolSource,
)
from .registry import CONNECTORS


@pytest.fixture(name="dynamic_connector")
def fixture_dynamic_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Mock, Mock]:
    """Register a dynamic provider with mocked discovery and execution."""
    discover = Mock(
        return_value=(
            ActionDefinition(
                name="FutureTool",
                description="Future",
                access=ActionAccess.WRITE,
                input_schema={"type": "object"},
            ),
        )
    )
    execute = Mock(return_value={"ok": True})
    connector = ConnectorDefinition(
        ref="example",
        tools=(),
        executors={},
        requires_credentials=True,
        tool_source=ConnectorToolSource(discover, execute),
    )
    monkeypatch.setitem(
        registry._CONNECTORS_BY_REF,  # pylint: disable=protected-access
        "example",
        connector,
    )
    return discover, execute


def test_connector_references_are_unique() -> None:
    """Connector references should be unique."""
    connector_refs = [connector.ref for connector in CONNECTORS]

    assert len(connector_refs) == len(set(connector_refs))


def test_connector_tool_names_are_unique() -> None:
    """Connector tool names should be unique."""
    tool_names = [name for connector in CONNECTORS for name in connector.executors]

    assert len(tool_names) == len(set(tool_names))


def test_dynamic_discovery_and_execution_are_separate(
    dynamic_connector: tuple[Mock, Mock],
) -> None:
    """Discover schemas and execute with bound credentials and the original name."""
    discover, execute = dynamic_connector
    usage_recorder = Mock()
    credentials: JSONObject = {"access_token": "bound-token"}
    assert registry.get_connector_ref("example_futuretool") == "example"
    assert registry.invoke_connector(
        "example__discover_tools", {}, usage_recorder, credentials, {}
    ) == [discover.return_value[0].tool("example")]
    assert registry.invoke_connector(
        "example__call_tool",
        {"name": "FutureTool", "arguments": {"value": 1}},
        usage_recorder,
        credentials,
        {},
    ) == {"ok": True}
    discover.assert_called_once_with(
        ConnectorExecutionContext(credentials, {}, usage_recorder)
    )
    execute.assert_called_once_with(
        {"value": 1},
        ConnectorExecutionContext(
            credentials, {}, usage_recorder, action_name="FutureTool"
        ),
    )


@pytest.mark.parametrize("arguments", [{}, {"name": "FutureTool", "arguments": []}])
def test_invalid_dynamic_calls_do_not_execute(
    dynamic_connector: tuple[Mock, Mock], arguments: JSONObject
) -> None:
    """Validate the internal call envelope before entering the provider executor."""
    _, execute = dynamic_connector
    with pytest.raises(ValueError):
        registry.invoke_connector("example__call_tool", arguments, Mock(), {}, {})
    execute.assert_not_called()
