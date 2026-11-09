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

Both plugins are published as `context-graph` in the [Memgraph marketplace](https://github.com/memgraph/marketplace), named `memgraph`, which lists every Memgraph plugin. Its entries point at these folders on this repository's default branch. The plugins call runtime adapters; graph connectors decide what to store.

This repository's own `context-graph-plugins` marketplace is the one setup used before. It stays for a while, so installs made with an older `memgraph-context-graph` keep updating; setup and update move them to the Memgraph marketplace. Don't point anything new at it:

| Runtime | Previous marketplace file |
|---|---|
| Codex | [`.agents/plugins/marketplace.json`](../../.agents/plugins/marketplace.json) |
| Claude Code | [`.claude-plugin/marketplace.json`](../../.claude-plugin/marketplace.json) |
