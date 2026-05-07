---
name: context-graph
description: Check, explain, and troubleshoot Agent Context Graph capture from Codex hooks into Memgraph-backed graph connectors.
---

# Context Graph

Use this skill when the user asks whether Codex activity is being captured, how Agent Context Graph hooks work, or why graph facts are missing.

## Model

Codex Plugin -> Codex Runtime Adapter -> Event Protocol -> Graph Connector -> Memgraph.

The plugin only installs runtime hook wiring. It must not assign graph meaning. Graph connectors decide which normalized events matter.

## Checks

1. Verify `agent-context-graph` is on `PATH`.
2. Verify `skills-graph` is installed in the same Python environment.
3. Verify Memgraph is reachable, usually at `bolt://localhost:7687`.
4. Verify the hook command accepts a minimal payload:

```bash
printf '{"hook_event_name":"Stop","session_id":"doctor"}' \
  | agent-context-graph hook run codex --connector skills-graph
```

Expected output:

```json
{"continue": true}
```

5. If skill usage is missing, inspect whether the session actually read or invoked a skill. Search/list results can be surfacing, not proven usage.
