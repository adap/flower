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
"""Tests for Notion MCP OAuth configuration."""

import traceback
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest

from flwr.supercore.typing import JSONObject

from ..oauth import OAuthFlow
from .definition import PROVIDER

_ACCESS_TOKEN = "notion-private-token-fixture"
_TOKEN_REQUEST = "flwr.supercore.task_process.connector.oauth.requests.post"


def _response(payload: object) -> Mock:
    """Return a minimal successful token response."""
    return Mock(status_code=200, json=Mock(return_value=payload))


def _flow() -> OAuthFlow:
    """Build the configured confidential Notion MCP client."""
    return OAuthFlow(
        PROVIDER,
        client_id="mcp-client",
        client_secret="mcp-secret",
        redirect_uri="https://example.com/callback",
    )


def test_notion_oauth_uses_mcp_endpoints_and_pkce() -> None:
    """Bind authorization and token exchange to the official MCP resource."""
    flow = _flow()
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
    assert parse_qs(url.query) == {
        "client_id": ["mcp-client"],
        "redirect_uri": ["https://example.com/callback"],
        "response_type": ["code"],
        "state": ["state"],
        "scope": ["default"],
        "resource": ["https://mcp.notion.com/mcp"],
        "code_challenge": ["challenge"],
        "code_challenge_method": ["S256"],
    }
    payload = {
        "access_token": "mcp-token",
        "refresh_token": "refresh-token",
        "expires_in": 3600,
        "token_type": "Bearer",
        "scope": "default",
        "user_id": "user-1",
        "workspace_id": "workspace-1",
        "workspace_name": "Engineering",
        "bot_id": "bot-1",
        "email_domain": "example.com",
        "workspace_icon": None,
        "future_field": "ignored",
    }
    with patch(_TOKEN_REQUEST, return_value=_response(payload)) as post:
        credentials, config = flow.exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
    post.assert_called_once_with(
        "https://mcp.notion.com/token",
        auth=None,
        headers={"Accept": "application/json"},
        data={
            "grant_type": "authorization_code",
            "code": "code",
            "redirect_uri": "https://example.com/callback",
            "client_id": "mcp-client",
            "client_secret": "mcp-secret",
            "code_verifier": "verifier",
            "resource": "https://mcp.notion.com/mcp",
        },
        timeout=30.0,
    )
    assert credentials == {
        "access_token": "mcp-token",
        "refresh_token": "refresh-token",
        "expires_in": 3600,
        "token_type": "Bearer",
    }
    assert config == {
        "user_id": "user-1",
        "workspace_id": "workspace-1",
        "workspace_name": "Engineering",
        "bot_id": "bot-1",
        "email_domain": "example.com",
        "display_name": "Notion · Engineering",
    }


def test_notion_oauth_requires_pkce_and_configured_redirect() -> None:
    """Reject missing PKCE parameters and callbacks outside the registered URI."""
    flow = _flow()
    with pytest.raises(ValueError, match="requires PKCE"):
        flow.build_authorization_url(
            redirect_uri="https://example.com/callback",
            state="state",
            pkce_challenge=None,
        )
    with patch(_TOKEN_REQUEST) as post:
        with pytest.raises(RuntimeError, match="exchange failed"):
            flow.exchange_code(
                code="code",
                redirect_uri="https://example.com/callback",
                pkce_verifier=None,
            )
        with pytest.raises(ValueError, match="redirect URI is not allowed"):
            flow.exchange_code(
                code="code",
                redirect_uri="https://other.example/callback",
                pkce_verifier="verifier",
            )
    post.assert_not_called()


@pytest.mark.parametrize(
    "payload",
    [
        {"access_token": "token", "token_type": "basic"},
        {"access_token": "token", "token_type": "bearer", "scope": "other"},
        {"error": "invalid_grant", "error_description": _ACCESS_TOKEN},
    ],
)
def test_notion_oauth_rejects_invalid_tokens(payload: JSONObject) -> None:
    """Reject unusable grants without exposing token response details."""
    with (
        patch(_TOKEN_REQUEST, return_value=_response(payload)),
        pytest.raises(RuntimeError) as error,
    ):
        _flow().exchange_code(
            code="code",
            redirect_uri="https://example.com/callback",
            pkce_verifier="verifier",
        )
    assert _ACCESS_TOKEN not in "".join(traceback.format_exception(error.value))
