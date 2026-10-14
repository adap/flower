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
"""Tests for GitHub MCP discovery, execution, and error details."""

import traceback
from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock, Mock, call, create_autospec, patch

import httpx
import pytest
from mcp import ClientSession, types
from mcp.shared.exceptions import McpError

from flwr.supercore.typing import JSONObject

from .. import registry
from .mcp import GitHubApiError

_ACCESS_TOKEN = "github-private-token-fixture"
_CREDENTIALS: JSONObject = {"access_token": _ACCESS_TOKEN}


@pytest.fixture(name="session")
def fixture_session() -> Iterator[Mock]:
    """Mock the MCP session and transport."""
    session = create_autospec(ClientSession, instance=True)
    session.__aenter__.return_value = session
    transport = MagicMock()
    transport.__aenter__.return_value = (Mock(), Mock(), Mock())
    with (
        patch("mcp.ClientSession", return_value=session),
        patch(
            "mcp.client.streamable_http.streamable_http_client", return_value=transport
        ),
    ):
        yield session


def test_paginated_discovery_and_execution(session: Mock) -> None:
    """Preserve remote schemas, names, arguments, and results without rediscovery."""
    schema = {"type": "object", "properties": {"perPage": {"type": "integer"}}}
    session.list_tools.side_effect = [
        types.ListToolsResult(
            tools=[
                types.Tool(name="issue_write", description="Write", inputSchema=schema)
            ],
            nextCursor="next",
        ),
        types.ListToolsResult(
            tools=[types.Tool(name="FutureTool", inputSchema=schema)]
        ),
    ]
    tools = cast(
        list[JSONObject],
        registry.invoke_connector(
            "github__discover_tools", {}, Mock(), _CREDENTIALS, {}
        ),
    )
    assert [tool["name"] for tool in tools] == [
        "github_issue_write",
        "github_FutureTool",
    ]
    assert tools[0]["description"] == "Write"
    assert tools[0]["parameters"] == schema
    session.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="done")],
        structuredContent={"count": 1},
    )
    arguments: JSONObject = {"perPage": 5, "data": {"items": [1, False, None]}}
    result = registry.invoke_connector(
        "github_FutureTool", arguments, Mock(), _CREDENTIALS, {}
    )
    assert result == session.call_tool.return_value.model_dump(
        mode="json", by_alias=True, exclude_none=True
    )
    session.call_tool.assert_awaited_once_with("FutureTool", arguments)
    assert session.list_tools.await_args_list == [
        call(params=None),
        call(params=types.PaginatedRequestParams(cursor="next")),
    ]


@pytest.mark.parametrize("failure", ["tool", "http", "protocol"])
def test_error_details_are_preserved_and_redacted(session: Mock, failure: str) -> None:
    """Forward tool and SDK diagnostics without leaking credentials in exceptions."""
    message = f"Denied {_ACCESS_TOKEN}"
    if failure == "tool":
        session.call_tool.return_value = types.CallToolResult(
            content=[types.TextContent(type="text", text=message)], isError=True
        )
    elif failure == "http":
        session.call_tool.side_effect = httpx.HTTPStatusError(
            message,
            request=httpx.Request("POST", "https://example.com"),
            response=httpx.Response(403),
        )
    else:
        session.call_tool.side_effect = ExceptionGroup(
            "MCP failed", [McpError(types.ErrorData(code=-32602, message=message))]
        )
    with pytest.raises(GitHubApiError) as error:
        registry.invoke_connector("github_FutureTool", {}, Mock(), _CREDENTIALS, {})
    assert error.value.code == ("tool_error" if failure == "tool" else "request_failed")
    assert error.value.message == "Denied [REDACTED]"
    assert _ACCESS_TOKEN not in "".join(traceback.format_exception(error.value))
    if failure == "http":
        assert error.value.status_code == 403
