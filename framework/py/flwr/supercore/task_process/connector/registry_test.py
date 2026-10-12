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


def test_connector_references_are_unique() -> None:
    """Connector references should be unique."""
    connector_refs = [connector.ref for connector in CONNECTORS]

    assert len(connector_refs) == len(set(connector_refs))


def test_connector_tool_names_are_unique() -> None:
    """Connector tool names should be unique."""
    tool_names = [name for connector in CONNECTORS for name in connector.executors]

    assert len(tool_names) == len(set(tool_names))


def test_dynamic_discovery_and_execution_are_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discover schemas and execute with bound credentials and the original name."""
    action = ActionDefinition(
        name="FutureTool",
        description="Future",
        access=ActionAccess.WRITE,
        input_schema={"type": "object"},
    )
    discover = Mock(return_value=(action,))
    execute = Mock(return_value={"ok": True})
    for ref in ("example", "example_nested"):
        monkeypatch.setitem(
            registry._CONNECTORS_BY_REF,  # pylint: disable=protected-access
            ref,
            ConnectorDefinition(
                ref=ref,
                tools=(),
                executors={},
                requires_credentials=True,
                tool_source=ConnectorToolSource(discover, execute),
            ),
        )
    assert registry.get_connector_ref("example_nested_read") == "example_nested"
    assert registry.get_connector_ref("example_nested_") == "example"
    assert registry.get_connector_ref("exampleish_read") == "exampleish_read"
    usage_recorder = Mock()
    credentials: JSONObject = {"access_token": "bound-token"}
    assert registry.invoke_connector(
        "example__discover_tools", {}, usage_recorder, credentials, {}
    ) == [action.tool("example")]
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
