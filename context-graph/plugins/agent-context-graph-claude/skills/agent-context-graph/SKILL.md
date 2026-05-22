---
description: Set up and check Claude Code context capture into Memgraph.
---

# Agent Context Graph

Use this skill when the user asks whether Claude Code activity is being captured, how Agent Context Graph hooks work, or why graph facts are missing.

## Model

Claude Code Plugin -> Claude Code Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph.

The plugin only installs runtime hook wiring. It must not assign graph meaning. Graph connectors decide which normalized events matter.

## Checks

For first-time setup, run the plugin bootstrap once. It delegates to the CLI bootstrap and falls back to `uvx` if `agent-context-graph` is not installed yet:

```bash
./scripts/bootstrap.sh
```

If bootstrap says Memgraph is not reachable, tell the user to start Memgraph:

```bash
docker run --rm -p 7687:7687 memgraph/memgraph
```

Run the single CLI doctor first. It checks the same Python environment that the hook command uses:

```bash
agent-context-graph doctor --runtime claude-code --connector skills-graph --connector actions-graph --connector sessions-graph
```

If `doctor` is not available, use the strict hook smoke:

```bash
printf '{"hook_event_name":"Stop","session_id":"doctor"}' \
  | agent-context-graph hook run claude-code --connector skills-graph --connector actions-graph --connector sessions-graph --strict
```

Do not check `skills_graph` with system `python3`; `agent-context-graph` may be installed in an isolated `uv tool` or `pipx` environment.

## Configuration

Connection settings live in `~/.config/context-graph/config.toml`. Hook subprocesses read from this file (not environment variables).

If `MEMGRAPH_URL`, `MEMGRAPH_USER`, `MEMGRAPH_PASSWORD`, or `MEMGRAPH_DATABASE` environment variables are set when bootstrap runs, they are automatically written to the config file. No manual `config set` needed in that case.

View current config:

```bash
agent-context-graph config show
```

Set values:

```bash
agent-context-graph config set identity.user_id "username"
agent-context-graph config set memgraph.url "neo4j://coordinator:7687"
agent-context-graph config set memgraph.user "username"
agent-context-graph config set memgraph.password
agent-context-graph config set memgraph.database "dbname"
```

For HA clusters, use the `neo4j://` scheme pointing to any coordinator. The driver handles routing and failover automatically:

```bash
agent-context-graph config set memgraph.url "neo4j://memgraph-coordinator-1:7687"
```

Supported keys: `identity.user_id`, `memgraph.url`, `memgraph.user`, `memgraph.password`, `memgraph.database`.

If skill usage is missing, inspect whether the session actually read or invoked a skill. Search/list results can be surfacing, not proven usage.
