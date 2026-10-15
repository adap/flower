# GitHub connector

The `github` connector discovers and executes tools from GitHub's remote MCP
server at `https://api.githubcopilot.com/mcp/`. Install `flwr[agent]` for the MCP
dependency. `definition.py` connects directly to discovery and execution in
`mcp.py`, following the Slack connector's structure.

## Authentication and private repositories

The current OAuth configuration expects a **GitHub App**, with credentials in
`FLWR_GITHUB_CLIENT_ID`, `FLWR_GITHUB_CLIENT_SECRET`, and
`FLWR_GITHUB_REDIRECT_URI`. GitHub Apps use configured permissions and selected
repositories instead of OAuth scopes, so the authorization request omits `scope`.

Private repositories are supported when the app is installed for those repositories,
the app has the permissions required by the tool, and the authorizing user has
access. For example, reading repository files requires the app's Contents read
permission; changing files requires Contents write permission. Organization
installation approval and SAML SSO requirements can also affect access.

A traditional **OAuth App** is different: a token with no scopes grants public
read access, and private repository access requires the `repo` scope. This
connector does not request that scope and rejects token responses containing
nonempty scopes. To use the current configuration for private repositories,
configure and install a GitHub App with the required repository permissions.

See GitHub's documentation on [user access tokens and app permissions](https://docs.github.com/en/apps/creating-github-apps/authenticating-with-a-github-app/authenticating-with-a-github-app-on-behalf-of-a-user)
and [OAuth scopes](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/scopes-for-oauth-apps).

## Discovery and execution

Discovery requests all MCP toolsets and follows pagination with repeated-cursor
protection. Tools retain their remote schemas and names; read-only annotations
classify read actions, and other actions default to write. The runtime restores
original tool-name casing when executing model calls.

Each discovery or call uses a task-local session. Execution sends `tools/call`
directly to avoid a catalog lookup after a successful write. Errors preserve
remote diagnostics with the bound access token redacted.
