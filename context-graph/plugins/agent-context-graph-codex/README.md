# Agent Context Graph for Codex

Codex runtime plugin for Agent Context Graph.

The plugin installs Codex lifecycle hooks that call:

```bash
agent-context-graph hook run codex --connector skills-graph --connector actions-graph
```

Flow:

```text
Codex Plugin -> Codex Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
```

The plugin is only deployment wiring. Agent Context Graph normalizes runtime events. Graph connectors decide what to persist.

## First Run

The plugin installs hook wiring, while the runtime package is installed by the CLI bootstrap:

```bash
./scripts/bootstrap.sh
```

Bootstrap expects `uv` and a reachable Memgraph instance. If Memgraph is not running, start it and rerun bootstrap:

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
```

`uv` manages Python for the tool. If uv-managed Python downloads are blocked in your environment, install Python 3.10+ and rerun bootstrap.

Bootstrap installs and verifies:

```bash
agent-context-graph bootstrap --runtime codex --connector skills-graph --connector actions-graph
```

## Prerequisites

- Memgraph running and reachable over Bolt.
- `uv` available on `PATH`.
- `agent-context-graph` available on `PATH` after bootstrap.

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

Then install or enable `context-graph` from the Codex plugin UI.

The Agent Context Graph skill is exposed as:

```text
context-graph:agent-context-graph
```

This is a Codex marketplace only. Claude Code uses the separate marketplace at `.claude-plugin/marketplace.json`.
