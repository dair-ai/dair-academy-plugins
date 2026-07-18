# Official X MCP setup

Canonical documentation: <https://docs.x.com/tools/mcp>

The official X API MCP server is:

```text
https://api.x.com/mcp
```

## App-only read access

Create an app in the X Developer Portal, copy its App-only Bearer token, and connect an MCP client to the hosted server with an `Authorization` header:

```json
{
  "mcpServers": {
    "xapi": {
      "url": "https://api.x.com/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_APP_ONLY_BEARER_TOKEN"
      }
    }
  }
}
```

This is the simplest option for a read-only intelligence feed. The token belongs in the user's secret store or local uncommitted configuration, never in the plugin.

## OAuth 2.0 user context with xurl

Use the official `xurl` bridge for user-context access and any future write operations:

```json
{
  "mcpServers": {
    "xapi": {
      "command": "npx",
      "args": ["-y", "@xdevplatform/xurl", "mcp", "https://api.x.com/mcp"],
      "env": {
        "CLIENT_ID": "YOUR_X_APP_CLIENT_ID",
        "CLIENT_SECRET": "YOUR_X_APP_CLIENT_SECRET"
      }
    }
  }
}
```

The user must enable OAuth 2.0 on the X app and register `http://localhost:8080/callback`, unless they set and register another `REDIRECT_URI`. The first run opens a browser for PKCE login. On headless machines, authenticate first with `xurl auth oauth2 --headless`.

The bridge caches and refreshes tokens in `~/.xurl`. Treat that directory as secret. Set an MCP startup timeout of at least 300 seconds because the first handshake may wait for browser login.

## Optional documentation MCP

The separate X documentation MCP server is:

```text
https://docs.x.com/mcp
```

It exposes documentation search and page retrieval. It is useful while developing or troubleshooting a skill but is not required to render the feed.

## Portability rules

- Use MCP tool discovery when exact tool names differ across clients.
- Expect read tools such as `get_users_by_usernames`, `get_users_posts`, `search_posts_all`, `get_posts_by_ids`, and `get_users_by_id`.
- Request only the fields needed for the artifact.
- Keep the workflow read-only.
- Never ask a user to paste a token into the artifact, prompt, issue, or chat transcript.
- If the client has no X MCP connection, explain the setup rather than falling back to scraping.
