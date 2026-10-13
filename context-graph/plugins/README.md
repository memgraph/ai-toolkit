# Context Graph Runtime Plugins

Runtime plugins install host-specific hook wiring for Memgraph Context Graph.

```text
Runtime Plugin -> Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

This directory contains plugin packages for multiple runtimes:

- `memgraph-context-graph-codex`: OpenAI Codex plugin, published as `context-graph`.
- `memgraph-context-graph-claude`: Claude Code plugin, published as `context-graph`.

GitHub Copilot CLI, Cursor, OpenCode, Antigravity CLI, and Grok Build are built-in Runtime
Registrations but do not have marketplace packages in this directory. Their
host-facing files are generated with `memgraph-context-graph hook init <runtime>`.

The platform-specific directory names stay explicit, while the marketplace-facing plugin name is short so skills appear as `context-graph:setup`.

## Marketplace Files

Codex and Claude Code use different marketplace schemas, so they are intentionally separate.

- Codex public marketplace: `.agents/plugins/marketplace.json`
- Claude Code public marketplace: `.claude-plugin/marketplace.json`

Do not add the Claude Code plugin to the Codex marketplace or the Codex plugin to the Claude Code marketplace. Each runtime validates a different plugin manifest directory.

Marketplace entries do not pin a ref. They use the repository default branch.
