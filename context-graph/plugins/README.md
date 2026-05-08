# Context Graph Runtime Plugins

Runtime plugins install host-specific hook wiring for Agent Context Graph.

```text
Runtime Plugin -> Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

This directory contains plugin packages for multiple runtimes:

- `agent-context-graph-codex`: OpenAI Codex plugin, published as `context-graph`.
- `agent-context-graph-claude`: Claude Code plugin, published as `context-graph`.

The platform-specific directory names stay explicit, while the marketplace-facing plugin name is short so skills appear as `context-graph:agent-context-graph`.

## Marketplace Files

Codex and Claude Code use different marketplace schemas, so they are intentionally separate.

- Codex public marketplace: `.agents/plugins/marketplace.json`
- Claude Code public marketplace: `.claude-plugin/marketplace.json`

Do not add the Claude Code plugin to the Codex marketplace or the Codex plugin to the Claude Code marketplace. Each runtime validates a different plugin manifest directory.

Marketplace entries do not pin a ref. They use the repository default branch.
