# context-graph-viewer

A read-only terminal viewer for your Context Graph. It shows what your Claude Code / Codex hooks captured, whether
capture is failing, where reconciliation stands, and what the graph learned, without Cypher or Memgraph Lab.

```bash
memgraph-context-graph ui
```

It reads the same config file as the hooks (`CONTEXT_GRAPH_CONFIG`, else `~/.config/context-graph/config.toml`),
polls Memgraph every few seconds, and never writes to the graph.

## Install

The viewer runs in the same tool environment as `memgraph-context-graph`. Add it to the `uv tool install` command
your plugin's bootstrap runs:

```bash
uv tool install "memgraph-context-graph[mcp]" \
  --with "skills-graph[memgraph-context-graph]" \
  --with "actions-graph[memgraph-context-graph]" \
  --with "sessions-graph[memgraph-context-graph]" \
  --with context-graph-viewer
```

`context-graph-viewer` also installs its own command, which takes the same flags as `memgraph-context-graph ui`.

## Views

| View | Reached by | Shows |
|---|---|---|
| **Dashboard** | start | Sessions in scope, the pipeline (reconcile and embedding queues, extraction backend, ontology version, failures, capture tokens), capture health (the local capture-failure log plus hints from the graph), and new entities per day. |
| **Session** | `enter` on a session | The timeline with subagents folded under the call that spawned them, each action's input and the resources it touched, and what the session produced (chunks, entities, memory files). |
| **Entities** | `k` on the dashboard, `g` on a session | Entities merged by name, by support (how many sessions and memory files ground them) or most recent, with their provenance. |
| **Neighbourhood** | `enter` on an entity | One entity's relations grouped by type, with the evidence behind each. `enter` on a neighbour moves there; `esc` walks back. |

## Keys

| Key | Does |
|---|---|
| `a` | Widen the scope: your sessions → plus unattributed ones → everyone's |
| `p` | Narrow every view to one project (working directory) |
| `/` | Filter the table |
| `1`–`4` | Jump to a pane; `f` makes it fill the screen |
| `v` | Neighbourhood: relation tree or flat table |
| `+` / `-` | Neighbourhood: raise or lower the confidence floor |
| `h` / `s` | Entities: show value types (dates, amounts) / sort by most recent |
| `w` | Enterprise: next space this login can read |
| `r` | Refresh now |
| `?` | All keys |
| `esc` / `q` | Back / quit |

## Scope

By default you see your own sessions: `identity.user_id` in Community, your Memgraph login in Enterprise. Sessions
captured before a user was configured belong to nobody (reconcile files them under an `anon-…` user), so if the
dashboard opens empty, press `a`.

## Development

Tests run against a real Memgraph and drive the app headless:

```bash
./scripts/dev-memgraph.sh test context-graph-viewer
```
