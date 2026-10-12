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
    ConnectorToolSource,
)
from .registry import CONNECTORS


@pytest.fixture(name="dynamic_connector")
def fixture_dynamic_connector(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ConnectorDefinition, Mock, Mock]:
    """Register a test provider without network access or mutable tool registration."""
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
        tools=({"name": "example_fixed"},),
        executors={"example_fixed": Mock(return_value="static")},
        requires_credentials=True,
        tool_source=ConnectorToolSource(discover, execute),
    )
    monkeypatch.setitem(
        registry._CONNECTORS_BY_REF,  # pylint: disable=protected-access
        "example",
        connector,
    )
    monkeypatch.setitem(
        registry._CONNECTORS_BY_TOOL,  # pylint: disable=protected-access
        "example_fixed",
        connector,
    )
    return connector, discover, execute


def test_connector_references_are_unique() -> None:
    """Connector references should be unique."""
    connector_refs = [connector.ref for connector in CONNECTORS]

    assert len(connector_refs) == len(set(connector_refs))


def test_connector_tool_names_are_unique() -> None:
    """Connector tool names should be unique."""
    tool_names = [name for connector in CONNECTORS for name in connector.executors]

    assert len(tool_names) == len(set(tool_names))


def test_dynamic_discovery_and_execution_are_separate(
    dynamic_connector: tuple[ConnectorDefinition, Mock, Mock],
) -> None:
    """Discover once, then call through the ordinary executor with fresh context."""
    connector, discover, execute = dynamic_connector
    credentials: JSONObject = {"access_token": "bound-token"}
    tools = registry.invoke_connector(
        "example__discover_tools", {}, Mock(), credentials, {}
    )
    assert tools == [
        {
            "type": "function",
            "name": "example_FutureTool",
            "description": "Future",
            "parameters": {"type": "object"},
            "strict": False,
        }
    ]
    for value in (1, 2):
        assert registry.invoke_connector(
            "example__call_tool",
            {"name": "FutureTool", "arguments": {"value": value}},
            Mock(),
            credentials,
            {},
        ) == {"ok": True}
    discover.assert_called_once()
    assert discover.call_args.args[0].credentials == credentials
    assert execute.call_args.args[0] == {"value": 2}
    assert execute.call_args.args[1].action_name == "FutureTool"
    assert execute.call_args.args[1].credentials == credentials
    assert connector.tools == ({"name": "example_fixed"},)
    assert set(connector.executors) == {"example_fixed"}


def test_dynamic_routing_preserves_static_precedence(
    dynamic_connector: tuple[ConnectorDefinition, Mock, Mock],
) -> None:
    """Preserve static executors and distinguish similarly named providers."""
    _, discover, execute = dynamic_connector
    assert registry.get_connector_ref("example_futuretool") == "example"
    assert registry.requires_connector_credentials("example_futuretool")
    assert registry.get_connector_call_tool("example_fixed") is None
    assert (
        registry.get_connector_ref("exampleish_futuretool") == "exampleish_futuretool"
    )
    assert registry.invoke_connector("example_fixed", {}, Mock(), {}, {}) == "static"
    discover.assert_not_called()
    execute.assert_not_called()
    with pytest.raises(RuntimeError, match="credentials are required"):
        registry.invoke_connector("example__discover_tools", {}, Mock())


@pytest.mark.parametrize("arguments", [{}, {"name": "FutureTool", "arguments": []}])
def test_invalid_dynamic_calls_do_not_execute(
    dynamic_connector: tuple[ConnectorDefinition, Mock, Mock], arguments: JSONObject
) -> None:
    """Validate the internal call envelope before entering the provider executor."""
    _, _, execute = dynamic_connector
    with pytest.raises(ValueError):
        registry.invoke_connector("example__call_tool", arguments, Mock(), {}, {})
    execute.assert_not_called()
