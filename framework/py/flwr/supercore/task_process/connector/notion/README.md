# Notion connector

The `notion` connector discovers and calls every tool advertised by Notion's
official remote MCP server at `https://mcp.notion.com/mcp`. Install Flower with
the optional `agent` dependencies to use the MCP connector.

```python
tools = session.connectors.tools(["notion"])
# Supply tools to the model, then execute its function calls:
output = session.connectors.call(tool_call)
```

Discovery preserves remote descriptions, input schemas, and tool names. The
existing model-facing naming convention prefixes and lowercases the remote name:
`notion-search` becomes `notion_notion-search`. The executor forwards the original
remote name and arguments unchanged. Read-only MCP annotations classify read
actions; other actions default to write. Both read and write tools are exposed.

Tool catalogs are discovered inside the connector worker using the bound account's
credentials. The AgentApp receives only schemas, cached for its agent runtime with
independent copies returned to callers. Use
`session.connectors.tools(["notion"], refresh=True)` to rediscover. The shared
`ConnectorToolSource` API executes calls using their original remote names without
fetching the catalog again. The shared connector definition is never changed with
an account's tools.

## OAuth setup

Notion MCP requires its own OAuth client and access token. Existing Notion REST
integration credentials do not work with the hosted MCP server. Register a
confidential client once for the deployment at `https://mcp.notion.com/register`,
using the same callback URI that Flower will allow:

```json
{
  "client_name": "Flower",
  "redirect_uris": ["https://your-flower.example/oauth/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "client_secret_post"
}
```

Send this JSON in a POST with `Content-Type: application/json`. Store the returned
`client_id` and `client_secret` in `FLWR_NOTION_CLIENT_ID` and
`FLWR_NOTION_CLIENT_SECRET`, and set `FLWR_NOTION_REDIRECT_URI` to the registered
callback URI. Production callbacks use HTTPS; HTTP is permitted for loopback
callbacks. Registration is a deployment setup step, following Flower's existing
environment-configured OAuth pattern.

Flower authorizes at `https://mcp.notion.com/authorize` and exchanges the code at
`https://mcp.notion.com/token`, with S256 PKCE, the `default` scope, and the MCP
resource indicator. Workspace and user identity fields from the token response
are saved in connection config; `workspace_name` supplies the connection label.
The saved access token is sent only to the MCP endpoint as a bearer token.
Refresh tokens and expiry information are stored by the existing OAuth flow;
automatic token refresh is not currently implemented. Reconnect after expiry.

See Notion's [MCP client documentation](https://developers.notion.com/guides/mcp/build-mcp-client)
and [authorization server metadata](https://mcp.notion.com/.well-known/oauth-authorization-server)
for registration and authentication details.

## Migration and responses

This replaces the former static REST tool catalog. Reconnect existing Notion
connections with MCP credentials and obtain tool names and schemas through
`session.connectors.tools(["notion"])`; old REST tool names and argument shapes
are no longer advertised.

Successful results preserve MCP `content`, optional `structuredContent`, and
`isError: false`. Errors use Flower's existing connector error channel, with
access tokens redacted from MCP and SDK messages.

The MCP SDK handles protocol framing, JSON/SSE responses, and session headers. Every
discovery or call opens and closes a task-local session, with a 120-second overall
timeout and 30-second HTTP and MCP request timeouts. There are no application-level
retries, including for write tools.

Tool calls send `tools/call` directly, so a completed write cannot be reported as
failed by the SDK's subsequent output-schema catalog lookup. Discovery stops with
`invalid_tools` if the server repeats a pagination cursor or cycles between cursors.
