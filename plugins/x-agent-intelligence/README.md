# X Agent Intelligence

Build a polished, local HTML intelligence feed from public AI and agent updates collected through the official X MCP server.

Version: 1.4.0

This plugin provides a vendor-neutral skill, X setup guide, an optional public starter source list, and a self-contained visual reference artifact. It does not include credentials, local state, a hosted service, a scheduler, or orchestration-specific configuration.

## Install

```text
/plugin marketplace add dair-ai/dair-academy-plugins
/plugin install x-agent-intelligence@dair-academy-plugins
```

## 1. Configure X MCP

Follow the official X MCP documentation: <https://docs.x.com/tools/mcp>

The official X API MCP server is `https://api.x.com/mcp`. The recommended local bridge is `xurl`, which performs OAuth 2.0 PKCE login and refreshes tokens locally. X also documents an app-only Bearer-token connection for read-only use.

The bridge configuration is conceptually:

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

Never commit real credentials, tokens, `~/.xurl`, or a populated MCP configuration file.

## 2. Ask your agent

After configuring the X MCP server, ask your agent:

```text
Use the x-agent-intelligence skill to build a self-contained local feed from my X MCP connection; ask for my source handles if needed, save feed.html, and validate it.
```

The skill asks for source handles when they are not supplied. To begin with the same public source mix as the reference feed, tell it to use `references/starter-sources.md`; you can edit or replace that list. The generated feed also includes a source-settings panel where you can add or remove handles later; those edits persist in browser local storage and take effect on the next refresh prompt.

The skill fetches only through your X MCP connection, builds the HTML locally, and leaves scheduling entirely to your own system. To refresh, run the same prompt again from any scheduler or agent session you choose.

## Included files

- `skills/x-agent-intelligence/SKILL.md` contains the agent workflow.
- `references/x-mcp-setup.md` explains authentication, client configuration, and portability.
- `references/starter-sources.md` provides an optional public starting point for source selection.
- `assets/reference-artifact.html` is the browser-ready design reference the skill reproduces.
- `agents/openai.yaml` is optional launcher metadata exposing the same one-line prompt as an agent shortcut.

The generated feed stores links and short summaries by default. It should not republish full post text or download third-party media unless the user has a suitable legal basis and wants that behavior.
