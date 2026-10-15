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
"""Discover and call Notion's official remote MCP tools."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, cast

import httpx

from flwr.supercore.typing import JSONObject, JSONValue

from ..http import ConnectorApiError

if TYPE_CHECKING:
    from mcp import ClientSession


class NotionApiError(ConnectorApiError):
    """Notion failure forwarded through the connector error channel."""

    provider = "Notion"


def request(
    name: str | None,
    arguments: JSONObject,
    credentials: JSONObject,
) -> JSONValue:
    """Discover tools or call one, forwarding errors without exposing the token."""
    token = credentials.get("access_token")
    if not isinstance(token, str) or not token:
        raise NotionApiError("invalid_credentials")
    try:
        return asyncio.run(_request(name, arguments, token))
    except Exception as ex:  # pylint: disable=broad-exception-caught
        while isinstance(ex, ExceptionGroup):
            ex = ex.exceptions[0]  # pylint: disable=no-member
        if isinstance(ex, NotionApiError):
            code, status, message = ex.code, ex.status_code, ex.message
        else:
            code = "request_failed"
            status = (
                ex.response.status_code
                if isinstance(ex, httpx.HTTPStatusError)
                else None
            )
            message = str(ex)
        error = NotionApiError(
            code, status, (message or "").replace(token, "[REDACTED]")
        )
    # Raise outside the handler so the original exception cannot expose the token.
    raise error


async def _request(
    name: str | None,
    arguments: JSONObject,
    token: str,
) -> JSONValue:
    """Initialize a task-local MCP session and discover or call remote tools."""
    from mcp import ClientSession, types  # pylint: disable=import-outside-toplevel
    from mcp.client.streamable_http import (  # pylint: disable=import-outside-toplevel
        streamable_http_client,
    )

    async with (
        asyncio.timeout(120),
        httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"},
            timeout=30,
            follow_redirects=False,
        ) as client,
        streamable_http_client("https://mcp.notion.com/mcp", http_client=client) as (
            read,
            write,
            _,
        ),
    ):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=30)
        ) as session:
            await session.initialize()
            if name is None:
                return await _list_tools(session)
            # Avoid call_tool's catalog fetch after executing a tool in a fresh session.
            result = await session.send_request(
                types.ClientRequest(
                    types.CallToolRequest(
                        params=types.CallToolRequestParams(
                            name=name, arguments=arguments
                        )
                    )
                ),
                types.CallToolResult,
            )
            if result.isError:
                raise NotionApiError(
                    "tool_error",
                    message=" ".join(
                        item.text
                        for item in result.content
                        if isinstance(item, types.TextContent)
                    ),
                )
            return cast(
                JSONObject,
                result.model_dump(mode="json", by_alias=True, exclude_none=True),
            )


async def _list_tools(session: ClientSession) -> JSONValue:
    """Read every page of the authenticated tool catalog."""
    tools: list[JSONObject] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        page = await session.list_tools(cursor=cursor)
        tools.extend(
            cast(
                JSONObject,
                tool.model_dump(mode="json", by_alias=True, exclude_none=True),
            )
            for tool in page.tools
        )
        cursor = page.nextCursor
        if not cursor:
            return cast(JSONValue, tools)
        if cursor in seen_cursors:
            raise NotionApiError("invalid_tools")
        seen_cursors.add(cursor)
