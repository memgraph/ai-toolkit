# Agent Context Graph for Claude Code

Claude Code runtime plugin for Agent Context Graph.

The plugin installs Claude Code lifecycle hooks that call:

```bash
agent-context-graph hook run claude-code --connector skills-graph
```

Flow:

```text
Claude Code Plugin -> Claude Code Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
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

## Git Marketplace

Register this repo as a Claude Code marketplace:

```text
/plugin marketplace add memgraph/ai-toolkit
```

CLI equivalent:

```bash
claude plugin marketplace add memgraph/ai-toolkit --sparse .claude-plugin
```

Install:

```text
/plugin install agent-context-graph-claude@context-graph-plugins
```

This is a Claude Code marketplace only. Codex uses the separate marketplace at `.agents/plugins/marketplace.json`.
