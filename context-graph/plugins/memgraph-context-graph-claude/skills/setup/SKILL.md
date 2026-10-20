---
name: setup
description: Set up and check Claude Code context capture into Memgraph.
---

# Memgraph Context Graph

Use this skill when the user asks whether Claude Code activity is being captured, how Memgraph Context Graph hooks work, or why graph facts are missing.

## Model

Claude Code Plugin -> Claude Code Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph.

The plugin only installs runtime hook wiring. It must not assign graph meaning. Graph connectors decide which normalized events matter.

## Checks

For first-time setup, run the plugin bootstrap once. It installs the CLI and runs `memgraph-context-graph setup claude-code --no-tui`, which finds or starts Memgraph, sets the user's identity and checks the hooks:

```bash
./scripts/bootstrap.sh
```

If setup reports a failure, show the user its "To fix" line; it is the exact command to run next.

Run the single CLI doctor first. It checks the same Python environment that the hook command uses:

```bash
memgraph-context-graph doctor --runtime claude-code
```

If `doctor` is not available, use the strict hook smoke:

```bash
printf '{"hook_event_name":"Stop","session_id":"doctor"}' \
  | memgraph-context-graph hook run claude-code --strict
```

Do not check `skills_graph` with system `python3`; `memgraph-context-graph` may be installed in an isolated `uv tool` or `pipx` environment.

## Configuration

Connection settings live in `~/.config/context-graph/config.toml`. Hook subprocesses read from this file (not environment variables).

setup writes `memgraph.url` and `identity.user_id`; other values (credentials, keys) are set with `config set`.

View current config:

```bash
memgraph-context-graph config show
```

Set values:

```bash
memgraph-context-graph config set identity.user_id "username"
memgraph-context-graph config set memgraph.url "neo4j://coordinator:7687"
memgraph-context-graph config set memgraph.user "username"
memgraph-context-graph config set memgraph.password
memgraph-context-graph config set memgraph.database "dbname"
```

For HA clusters, use the `neo4j://` scheme pointing to any coordinator. The driver handles routing and failover automatically:

```bash
memgraph-context-graph config set memgraph.url "neo4j://memgraph-coordinator-1:7687"
```

Supported keys: `identity.user_id`, `memgraph.url`, `memgraph.user`, `memgraph.password`, `memgraph.database`.

If skill usage is missing, inspect whether the session actually read or invoked a skill. Search/list results can be surfacing, not proven usage.
