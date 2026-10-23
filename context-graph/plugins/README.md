# Context Graph plugins

Plugins connect Claude Code and Codex to [Context Graph](../README.md), installing lifecycle hooks and the MCP server for memory and recall.

## Install

The [Context Graph installer](../README.md#getting-started) installs the plugin and its Python dependencies together. For manual marketplace installation, use the guide for your agent:

| Agent | Plugin guide |
|---|---|
| Claude Code | [Install and configure](memgraph-context-graph-claude/) |
| Codex | [Install, trust hooks, and configure](memgraph-context-graph-codex/) |

Copilot CLI, Cursor, OpenCode, Antigravity CLI, and Grok Build use generated project wiring rather than marketplace plugins. Run `memgraph-context-graph setup <runtime>`; see [runtime setup](../memgraph-context-graph/docs/runtime-setup.md).

## For plugin maintainers

Both plugins are published as `context-graph` in the `context-graph-plugins` marketplace. They call runtime adapters; graph connectors decide what to store.

The marketplace files use different schemas:

| Runtime | Marketplace file |
|---|---|
| Codex | [`.agents/plugins/marketplace.json`](../../.agents/plugins/marketplace.json) |
| Claude Code | [`.claude-plugin/marketplace.json`](../../.claude-plugin/marketplace.json) |

Keep each plugin in its own runtime's marketplace. Entries follow the repository's default branch rather than a pinned ref.
