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
"""Tests for Slack MCP discovery and execution."""

from typing import cast
from unittest.mock import AsyncMock, Mock, patch

from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from flwr.supercore.typing import JSONObject

from .. import registry
from . import mcp as slack_mcp


def test_mcp_discovery_and_execution() -> None:
    """Discover all Slack tools and execute former static names through MCP."""
    session = AsyncMock()
    session.list_tools.side_effect = [
        ListToolsResult(
            tools=[
                Tool(name="slack_search_public", inputSchema={"type": "object"}),
                Tool(name="slack_read_channel", inputSchema={"type": "object"}),
            ],
            nextCursor="next",
        ),
        ListToolsResult(tools=[]),
    ]
    session.send_request.return_value = CallToolResult(
        content=[TextContent(type="text", text="done")]
    )
    credentials: JSONObject = {"access_token": "xoxp-secret"}
    assert not registry.get_connector_tools("slack")
    assert registry.is_dynamic_connector_tool("slack_search_public")
    with patch.object(slack_mcp, "_session") as open_session:
        open_session.return_value.__aenter__.return_value = session
        tools = registry.invoke_connector(
            "slack__discover_tools", {}, Mock(), credentials, {}
        )
        assert isinstance(tools, list)
        assert [cast(JSONObject, tool)["name"] for tool in tools] == [
            "slack_search_public",
            "slack_read_channel",
        ]
        result = registry.invoke_connector(
            "slack_search_public",
            {"query": "release"},
            Mock(),
            credentials,
            {},
        )
    assert session.list_tools.await_count == 2
    assert session.list_tools.await_args.kwargs["params"].cursor == "next"
    request = session.send_request.await_args.args[0].root
    assert request.params.name == "slack_search_public"
    assert request.params.arguments == {"query": "release"}
    assert isinstance(result, dict)
    assert result["content"] == [{"type": "text", "text": "done"}]
