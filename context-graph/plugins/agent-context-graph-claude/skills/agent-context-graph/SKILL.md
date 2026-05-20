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

If skill usage is missing, inspect whether the session actually read or invoked a skill. Search/list results can be surfacing, not proven usage.
