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
"""Essential tests for Notion MCP and OAuth."""

import traceback
from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
from mcp import ClientSession, types

from flwr.supercore.typing import JSONObject

from .. import registry
from ..oauth import OAuthFlow
from .definition import PROVIDER
from .mcp import NotionApiError

_CREDENTIALS: JSONObject = {"access_token": "notion-private-token"}


@pytest.fixture(name="send_request")
def fixture_send_request() -> Iterator[AsyncMock]:
    """Keep real SDK sessions while mocking transport and protocol requests."""
    send_request = AsyncMock()
    transport = MagicMock()
    transport.__aenter__.return_value = (Mock(), Mock(), Mock())
    with (
        patch.object(ClientSession, "send_request", send_request),
        patch.object(ClientSession, "initialize", AsyncMock()),
        patch.object(
            ClientSession,
            "__aenter__",
            autospec=True,
            side_effect=lambda client: client,
        ),
        patch.object(ClientSession, "__aexit__", AsyncMock(return_value=False)),
        patch(
            "mcp.client.streamable_http.streamable_http_client", return_value=transport
        ),
    ):
        yield send_request


def test_mcp_discovery_and_execution(send_request: AsyncMock) -> None:
    """Preserve paginated schemas and avoid catalog fetches after successful writes."""
    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    send_request.side_effect = [
        types.ListToolsResult(
            tools=[
                types.Tool(name="FutureTool", description="Write", inputSchema=schema)
            ],
            nextCursor="next",
        ),
        types.ListToolsResult(
            tools=[types.Tool(name="notion-search", inputSchema=schema)]
        ),
    ]
    tools = cast(
        list[JSONObject],
        registry.invoke_connector(
            "notion__discover_tools", {}, Mock(), _CREDENTIALS, {}
        ),
    )
    assert [tool["name"] for tool in tools] == [
        "notion_FutureTool",
        "notion_notion-search",
    ]
    assert tools[0]["description"] == "Write"
    assert tools[0]["parameters"] == schema
    assert send_request.await_count == 2
    assert send_request.await_args_list[1].args[0].root.params.cursor == "next"

    output = types.CallToolResult(
        content=[types.TextContent(type="text", text="done")],
        structuredContent={"id": "page"},
    )
    send_request.reset_mock(side_effect=True)
    send_request.side_effect = [output, RuntimeError("Catalog unavailable")]
    arguments: JSONObject = {"title": "New page"}
    result = registry.invoke_connector(
        "notion_FutureTool", arguments, Mock(), _CREDENTIALS, {}
    )
    assert result == output.model_dump(mode="json", by_alias=True, exclude_none=True)
    send_request.assert_awaited_once()
    assert send_request.await_args is not None
    request = send_request.await_args.args[0].root
    assert request.method == "tools/call"
    assert request.params.name == "FutureTool"
    assert request.params.arguments == arguments


@pytest.mark.parametrize("cursors", [("next", "next"), ("first", "second", "first")])
def test_mcp_rejects_repeated_cursors(
    send_request: AsyncMock, cursors: tuple[str, ...]
) -> None:
    """Reject repeated and cycling cursors before fetching the same page again."""
    send_request.side_effect = [
        types.ListToolsResult(tools=[], nextCursor=cursor) for cursor in cursors
    ]
    with pytest.raises(NotionApiError, match="invalid_tools"):
        registry.invoke_connector(
            "notion__discover_tools", {}, Mock(), _CREDENTIALS, {}
        )
    assert send_request.await_count == len(cursors)


def test_mcp_preserves_redacted_error_details(send_request: AsyncMock) -> None:
    """Retain remote diagnostics without exposing the token in tracebacks."""
    send_request.return_value = types.CallToolResult(
        isError=True,
        content=[types.TextContent(type="text", text="Denied notion-private-token")],
    )
    with pytest.raises(NotionApiError) as error:
        registry.invoke_connector("notion_FutureTool", {}, Mock(), _CREDENTIALS, {})
    assert error.value.code == "tool_error"
    assert error.value.message == "Denied [REDACTED]"
    assert "notion-private-token" not in "".join(
        traceback.format_exception(error.value)
    )


def test_oauth_uses_mcp_endpoints_pkce_and_resource() -> None:
    """Use Notion's MCP grant and bind the code exchange to its resource."""
    flow = OAuthFlow(
        PROVIDER,
        client_id="client",
        client_secret="secret",
        redirect_uri="https://example.com/callback",
    )
    url = urlparse(
        flow.build_authorization_url(
            redirect_uri="https://example.com/callback",
            state="state",
            pkce_challenge="challenge",
        )
    )
    assert (
        f"{url.scheme}://{url.netloc}{url.path}" == "https://mcp.notion.com/authorize"
    )
    query = parse_qs(url.query)
    assert query["resource"] == ["https://mcp.notion.com/mcp"]
    assert query["scope"] == ["default"]
    assert query["code_challenge"] == ["challenge"]
    assert query["code_challenge_method"] == ["S256"]
    payload = {
        "access_token": "token",
        "refresh_token": "refresh",
        "token_type": "bearer",
        "scope": "default",
        "workspace_name": "Engineering",
    }
    with patch(
        "flwr.supercore.task_process.connector.oauth.requests.post",
        return_value=Mock(status_code=200, json=Mock(return_value=payload)),
    ) as post:
        credentials, config = flow.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
    post.assert_called_once()
    assert post.call_args.args == ("https://mcp.notion.com/token",)
    data = post.call_args.kwargs["data"]
    assert data["resource"] == "https://mcp.notion.com/mcp"
    assert data["code_verifier"] == "verifier"
    assert data["client_secret"] == "secret"
    assert credentials["access_token"] == "token"
    assert credentials["refresh_token"] == "refresh"
    assert config["display_name"] == "Notion · Engineering"
