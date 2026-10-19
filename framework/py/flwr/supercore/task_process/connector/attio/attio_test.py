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
"""Tests for Attio MCP discovery, execution, and OAuth."""

import traceback
from collections.abc import Iterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from mcp import ClientSession, types
from mcp.shared.exceptions import McpError

from flwr.supercore.typing import JSONObject, JSONValue

from .. import registry
from ..definition import ActionAccess, ConnectorExecutionContext
from ..oauth import OAuthFlow
from .definition import ATTIO_CONNECTOR_REF, CONNECTOR, PROVIDER
from .mcp import AttioApiError, discover_tools, execute_tool

_ACCESS_TOKEN = "attio-private-token-fixture"
_CREDENTIALS: JSONObject = {"access_token": _ACCESS_TOKEN}
_REDIRECT_URI = "https://client.example/oauth/attio"
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"


@pytest.fixture(name="send_request")
def fixture_send_request() -> Iterator[AsyncMock]:
    """Use fresh SDK sessions with mocked transport and protocol requests."""
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


def _context() -> ConnectorExecutionContext:
    return ConnectorExecutionContext(
        credentials=_CREDENTIALS, config={}, usage_recorder=Mock()
    )


def test_paginated_discovery_and_execution(send_request: AsyncMock) -> None:
    """Preserve native names, schemas, and results without rediscovery on calls."""
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    send_request.side_effect = [
        types.ListToolsResult(
            tools=[
                types.Tool(
                    name="search-records",
                    description="Search records",
                    inputSchema=schema,
                )
            ],
            nextCursor="next",
        ),
        types.ListToolsResult(
            tools=[types.Tool(name="create-note", inputSchema=schema)]
        ),
    ]
    assert not registry.get_connector_tools(ATTIO_CONNECTOR_REF)
    assert CONNECTOR.executors == {}
    assert registry.get_connector_discovery_tool(ATTIO_CONNECTOR_REF) == (
        "attio__discover_tools"
    )
    tools = cast(
        list[JSONObject],
        registry.invoke_connector(
            "attio__discover_tools", {}, Mock(), _CREDENTIALS, {}
        ),
    )
    assert [tool["name"] for tool in tools] == [
        "attio_search-records",
        "attio_create-note",
    ]
    assert tools[0]["description"] == "Search records"
    assert tools[0]["parameters"] == schema
    assert send_request.await_count == 2
    assert send_request.await_args_list[1].args[0].root.params.cursor == "next"
    output = types.CallToolResult(
        content=[
            types.TextContent(type="text", text="done"),
            types.ImageContent(type="image", data="eA==", mimeType="image/png"),
        ],
        structuredContent={"count": 1},
    )
    send_request.reset_mock(side_effect=True)
    send_request.side_effect = [output, RuntimeError("Catalog unavailable")]
    arguments: JSONObject = {"query": "Flower", "data": {"items": [1, False, None]}}
    result = registry.invoke_connector(
        "attio_search-records", arguments, Mock(), _CREDENTIALS, {}
    )
    assert result == output.model_dump(mode="json", by_alias=True, exclude_none=True)
    send_request.assert_awaited_once_with(
        types.ClientRequest(
            types.CallToolRequest(
                params=types.CallToolRequestParams(
                    name="search-records", arguments=arguments
                )
            )
        ),
        types.CallToolResult,
    )


def test_discovery_access_and_authenticated_transport(send_request: AsyncMock) -> None:
    """Use the bound token and classify tools conservatively from annotations."""
    send_request.return_value = types.ListToolsResult(
        tools=[
            types.Tool(
                name="whoami",
                inputSchema={},
                annotations=types.ToolAnnotations(readOnlyHint=True),
            ),
            types.Tool(
                name="create-note",
                inputSchema={},
                annotations=types.ToolAnnotations(readOnlyHint=False),
            ),
            types.Tool(name="future-tool", inputSchema={}),
        ]
    )
    with patch(
        "flwr.supercore.task_process.connector.attio.mcp.httpx.AsyncClient",
        wraps=httpx.AsyncClient,
    ) as client:
        actions = discover_tools(_context())
    assert [action.access for action in actions] == [
        ActionAccess.READ,
        ActionAccess.WRITE,
        ActionAccess.WRITE,
    ]
    assert client.call_args.kwargs["headers"] == {
        "Authorization": f"Bearer {_ACCESS_TOKEN}"
    }
    assert client.call_args.kwargs["follow_redirects"] is False


@pytest.mark.parametrize("cursors", [("next", "next"), ("first", "second", "first")])
def test_discovery_rejects_repeated_cursors(
    send_request: AsyncMock, cursors: tuple[str, ...]
) -> None:
    """Stop at the first repeated cursor instead of looping indefinitely."""
    send_request.side_effect = [
        types.ListToolsResult(tools=[], nextCursor=cursor) for cursor in cursors
    ]
    with pytest.raises(AttioApiError) as error:
        discover_tools(_context())
    assert error.value.code == "invalid_tools"
    assert send_request.await_count == len(cursors)


@pytest.mark.parametrize("failure", ["tool", "http", "protocol"])
def test_error_details_are_preserved_and_redacted(
    send_request: AsyncMock, failure: str
) -> None:
    """Forward diagnostics without leaking credentials through exceptions."""
    message = f"Denied {_ACCESS_TOKEN}"
    if failure == "tool":
        send_request.return_value = types.CallToolResult(
            content=[types.TextContent(type="text", text=message)], isError=True
        )
    elif failure == "http":
        send_request.side_effect = httpx.HTTPStatusError(
            message,
            request=httpx.Request("POST", "https://mcp.attio.com/mcp"),
            response=httpx.Response(403),
        )
    else:
        send_request.side_effect = ExceptionGroup(
            "MCP failed", [McpError(types.ErrorData(code=-32602, message=message))]
        )
    with pytest.raises(AttioApiError) as error:
        registry.invoke_connector("attio_whoami", {}, Mock(), _CREDENTIALS, {})
    assert error.value.code == ("tool_error" if failure == "tool" else "request_failed")
    assert error.value.message == "Denied [REDACTED]"
    assert _ACCESS_TOKEN not in "".join(traceback.format_exception(error.value))
    if failure == "http":
        assert error.value.status_code == 403


@pytest.mark.parametrize("token", [None, "", 42])
def test_invalid_credentials(token: JSONValue) -> None:
    """Reject missing or invalid access tokens before opening a connection."""
    context = ConnectorExecutionContext(
        credentials={"access_token": token}, config={}, usage_recorder=Mock()
    )
    with pytest.raises(AttioApiError) as error:
        discover_tools(context)
    assert error.value.code == "invalid_credentials"


def test_execution_requires_action_name() -> None:
    """Reject calls without a remote tool name."""
    with pytest.raises(AttioApiError) as error:
        execute_tool({}, _context())
    assert error.value.code == "invalid_tool"


def test_oauth_uses_mcp_endpoints_and_pkce() -> None:
    """Use Attio's MCP OAuth flow and refresh tokens without REST identity calls."""
    flow = OAuthFlow(
        PROVIDER,
        client_id="client-id",
        client_secret="client-secret",
        redirect_uri=_REDIRECT_URI,
    )
    url = urlparse(
        flow.build_authorization_url(
            redirect_uri=_REDIRECT_URI,
            state="oauth-state",
            pkce_challenge="pkce-challenge",
        )
    )
    assert f"{url.scheme}://{url.netloc}{url.path}" == (
        "https://app.attio.com/oidc/authorize"
    )
    assert parse_qs(url.query) == {
        "client_id": ["client-id"],
        "redirect_uri": [_REDIRECT_URI],
        "response_type": ["code"],
        "state": ["oauth-state"],
        "scope": ["mcp offline_access"],
        "code_challenge": ["pkce-challenge"],
        "code_challenge_method": ["S256"],
    }
    with pytest.raises(ValueError, match="PKCE"):
        flow.build_authorization_url(
            redirect_uri=_REDIRECT_URI, state="oauth-state", pkce_challenge=None
        )
    with pytest.raises(ValueError, match="redirect URI"):
        flow.resolve_redirect_uri("https://attacker.example/callback")
    with pytest.raises(RuntimeError, match="exchange failed"):
        flow.exchange_code(code="code", redirect_uri=_REDIRECT_URI, pkce_verifier=None)

    response = Mock(status_code=200)
    response.json.return_value = {
        "access_token": "mcp-access",
        "token_type": "Bearer",
        "refresh_token": "mcp-refresh",
        "expires_in": 3600,
        "scope": "mcp offline_access",
    }
    with (
        patch("flwr.supercore.task_process.connector.oauth.time", return_value=1000),
        patch(_TOKEN_REQUEST, return_value=response) as post,
        patch(
            "flwr.supercore.task_process.connector.oauth.requests.request"
        ) as identity,
    ):
        credentials, config = flow.exchange_code(
            code="authorization-code",
            redirect_uri=_REDIRECT_URI,
            pkce_verifier="pkce-verifier",
        )
        post.assert_called_once()
        assert post.call_args.args == ("https://app.attio.com/oidc/token",)
        assert post.call_args.kwargs["data"] == {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "grant_type": "authorization_code",
            "code": "authorization-code",
            "redirect_uri": _REDIRECT_URI,
            "code_verifier": "pkce-verifier",
        }
        assert credentials == {
            "access_token": "mcp-access",
            "token_type": "Bearer",
            "refresh_token": "mcp-refresh",
            "expires_in": 3600,
            "expires_at": 4600,
        }
        assert not config
        identity.assert_not_called()
        response.json.return_value = {
            "access_token": "refreshed-access",
            "token_type": "Bearer",
            "refresh_token": "rotated-refresh",
            "expires_in": 3600,
        }
        credentials["expires_at"] = 0
        refreshed = flow.refresh_credentials(credentials)
        assert refreshed["access_token"] == "refreshed-access"
        assert refreshed["refresh_token"] == "rotated-refresh"
        assert post.call_args.args == ("https://app.attio.com/oidc/token",)
        assert post.call_args.kwargs["data"] == {
            "client_id": "client-id",
            "client_secret": "client-secret",
            "grant_type": "refresh_token",
            "refresh_token": "mcp-refresh",
        }
