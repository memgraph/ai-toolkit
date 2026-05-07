---
name: agent-context-graph
description: Check, explain, and troubleshoot Agent Context Graph capture from Codex hooks into Memgraph-backed graph connectors.
---

# Agent Context Graph

Use this skill when the user asks whether Codex activity is being captured, how Agent Context Graph hooks work, or why graph facts are missing.

## Model

Codex Plugin -> Codex Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph.

The plugin only installs runtime hook wiring. It must not assign graph meaning. Graph connectors decide which normalized events matter.

## Checks

Run the single CLI doctor first. It checks the same Python environment that the hook command uses:

```bash
agent-context-graph doctor --runtime codex --connector skills-graph
```

If `doctor` is not available, use the strict hook smoke:

```bash
printf '{"hook_event_name":"Stop","session_id":"doctor"}' \
  | agent-context-graph hook run codex --connector skills-graph --strict
```

Do not check `skills_graph` with system `python3`; `agent-context-graph` may be installed in an isolated `uv tool` or `pipx` environment.

If skill usage is missing, inspect whether the session actually read or invoked a skill. Search/list results can be surfacing, not proven usage.
