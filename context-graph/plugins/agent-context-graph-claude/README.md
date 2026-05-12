# Agent Context Graph for Claude Code

Claude Code runtime plugin for Agent Context Graph.

The plugin installs Claude Code lifecycle hooks that call:

```bash
agent-context-graph hook run claude-code --connector skills-graph --connector actions-graph
```

Flow:

```text
Claude Code Plugin -> Claude Code Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph
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
agent-context-graph bootstrap --runtime claude-code --connector skills-graph --connector actions-graph
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
/plugin install context-graph@context-graph-plugins
```

The Agent Context Graph skill is exposed as:

```text
/context-graph:agent-context-graph
```

This is a Claude Code marketplace only. Codex uses the separate marketplace at `.agents/plugins/marketplace.json`.
