# Agent Context Graph for Codex

Codex runtime plugin for Agent Context Graph.

The plugin installs Codex lifecycle hooks that call:

```bash
agent-context-graph hook run codex --connector skills-graph
```

Flow:

```text
Codex Plugin -> Codex Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

The plugin is only deployment wiring. Agent Context Graph normalizes runtime events. Graph connectors decide what to persist.

## Prerequisites

- Memgraph running and reachable over Bolt.
- `agent-context-graph` available on `PATH`.
- `skills-graph[agent-context-graph]` installed in the same Python environment.

For a global user install:

```bash
uv tool install agent-context-graph --with "skills-graph[agent-context-graph]"
```

## Local Test

From the plugin directory:

```bash
./scripts/doctor.sh
```

## Global Marketplace

This repo exposes a public Git-backed marketplace at:

```text
.agents/plugins/marketplace.json
```

Register the marketplace from GitHub:

```bash
codex plugin marketplace add memgraph/ai-toolkit --sparse .agents/plugins
```

Then install or enable `agent-context-graph-codex` from the Codex plugin UI.

This is a Codex marketplace only. Claude Code uses the separate marketplace at `.claude-plugin/marketplace.json`.
