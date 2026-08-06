# Sessions Graph

Sessions Graph is the [Context Graph](../CONTEXT-MAP.md) component for session context and cross-session recall. It is the **authority on `(:Session)` nodes** in the Context Graph family. It stores free-form text assertions — called **Memories** — written explicitly by agents, and makes them searchable in future sessions.

Requires **Memgraph ≥ 3.6** (text search is stable from that release).

## Installation

```bash
pip install sessions-graph
```

To use with Agent Context Graph:

```bash
pip install sessions-graph[agent-context-graph]
```

To use [session reconciliation](#session-reconciliation) (entity extraction from session content):

```bash
pip install sessions-graph[reconciliation]
```

## Quick start

```python
from sessions_graph import SessionsGraph

graph = SessionsGraph()  # connects via MEMGRAPH_URL / MEMGRAPH_USER / MEMGRAPH_PASSWORD env vars
graph.setup()  # creates constraints and the text index (run once)

# Write a memory
mem = graph.save_memory(
    user_id="alice",
    content="Prefers Python over TypeScript",
    session_id="s-abc123",  # optional — links memory to a session for provenance
)

# Retrieve all memories for a user
memories = graph.get_memories("alice")

# Search memories by content (full-text, powered by Tantivy)
results = graph.search_memories("alice", "Python")

# Update or delete
graph.update_memory(mem.memory_id, "Prefers Python, especially for data tooling")
graph.delete_memory(mem.memory_id)
```

## Integration with Agent Context Graph

Wire the `SessionsGraphConnector` into an `AgentLink` to get automatic session provenance — the connector tracks the active `session_id` and `user_id` from `SessionStartEvent` so you can reference them when saving memories.

```python
from sessions_graph import SessionsGraph
from sessions_graph.connector import SessionsGraphConnector
from agent_context_graph import AgentLink
from agent_context_graph.adapters.claude import ClaudeAdapter

graph = SessionsGraph()
graph.setup()

connector = SessionsGraphConnector(graph)
link = AgentLink()
link.add_connector(connector)

adapter = ClaudeAdapter(
    link,
    session_id="s-abc123",
    session_kwargs={"user_id": "alice"},
)

# During the session, save memories via the Python API:
graph.save_memory(
    user_id=connector.active_user_id,
    content="User works primarily in the ai-toolkit repository",
    session_id=connector.active_session_id,
)
```

## Graph schema

```
(:User {user_id})
    ├─[:HAS_MEMORY]──▶ (:Memory {memory_id, user_id, content, created_at})
    │                          ▲                        │
    │            [:PRODUCED_MEMORY]              [:HAS_CHUNK]
    │                          │                        ▼
    └─[:HAD_SESSION]─▶ (:Session {session_id,   (:Chunk {hash, text})
                                  reconciliation_status,     ▲
                                  reconciled_at})  [:HAS_CHUNK]
                              │                        │
                        [:HAS_ACTION]                  │
                              ▼                        │
                        (:Action) ─────────────────────┘
                              │
                                              (:Entity)-[:MENTIONED_IN]->(:Chunk)
```

`(:User)-[:HAD_SESSION]->(:Session)` is written by `SessionsGraphConnector` on
session start; it is the join key other Context Graph components hang off of.

`(:Action)` is owned by [Actions Graph](../actions-graph/); `(:Chunk)` and the
extracted entity nodes are owned by
[unstructured2graph](../../unstructured2graph/). See [Session
reconciliation](#session-reconciliation) below for how they get linked.

## Text search

Sessions Graph uses [Memgraph text search](https://memgraph.com/docs/querying/text-search) (powered by Tantivy) for `search_memories`. The text index is created on `setup()`:

```cypher
CREATE TEXT INDEX memory_content_index ON :Memory(content);
```

Searches run as:

```cypher
CALL text_search.search_all('memory_content_index', 'Python')
YIELD node AS m, score
WHERE m.user_id = 'alice'
RETURN m.content, score
ORDER BY score DESC
LIMIT 10;
```

The query string follows [Tantivy query syntax](https://docs.rs/tantivy/latest/tantivy/query/struct.QueryParser.html).

## Session reconciliation

A session's Actions Graph content (Messages, ToolCalls, ToolResults) and
Memories are mostly opaque text today. Session reconciliation runs that content
through [unstructured2graph](../../unstructured2graph/)'s chunk + LightRAG
entity-extraction pipeline, turning it into queryable graph entities linked
back to the session that produced them — see
[`CONTEXT.md`](./CONTEXT.md#language) for the **Session Reconciliation** /
**Reconcilable Content** / **Reconciliation Status** terminology.

This requires the `sessions-graph[reconciliation]` extra and an LLM API key
(`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) for LightRAG — see the
[lightrag-memgraph README](../../integrations/lightrag-memgraph/README.md).

**Reconciliation never runs inside the `SESSION_END` hook itself.** LightRAG
entity extraction is LLM-backed and slow, and hook runtimes (Claude Code,
Codex) enforce a timeout on hook commands. Instead:

- On `SESSION_END`, `SessionsGraphConnector` cheaply marks the session
  `reconciliation_status = 'pending'` — no LLM calls, safe inside the hook.
- The actual reconciliation run happens out-of-band, via the CLI:

  ```bash
  # Reconcile one session
  sessions-graph reconcile --session s-abc123

  # Sweep every session still marked 'pending' (e.g. from cron)
  sessions-graph reconcile --pending --limit 50

  # Optional: override LightRAG's working dir (default ./lightrag_storage)
  sessions-graph reconcile --pending --working-dir ./lightrag_storage
  ```

  The CLI runs with **`enforce_ontology=True`**: extracted entities get real
  type labels (`:Person`, `:Organization`, …) gated by unstructured2graph's
  default ontology, and anything outside it is kept but flagged
  `ontology_conformant = false`. See [entity typing](../../unstructured2graph/README.md#entity-typing--ontology).

- Or, if you want it triggered automatically without a manual/cron step, opt
  in to a **best-effort detached background process** spawned right after a
  session ends: pass `SessionsGraphConnector(graph, auto_reconcile=True)`, or set
  `SESSIONS_GRAPH_AUTO_RECONCILE=1` in the environment the connector is
  constructed in (this is what hook-based runtimes read, since they don't
  expose a constructor kwarg for it). This is fire-and-forget — if the
  process dies before finishing (machine sleep, crash), the session stays
  `pending` and `sessions-graph reconcile --pending` is the reliable backfill.

Programmatically:

```python
from sessions_graph import SessionsGraph
from lightrag_memgraph import MemgraphLightRAGWrapper

graph = SessionsGraph()
graph.setup()

lightrag_wrapper = MemgraphLightRAGWrapper()
await lightrag_wrapper.initialize(working_dir="./lightrag_storage")

summary = await graph.reconcile_session(
    "s-abc123",
    lightrag_wrapper=lightrag_wrapper,
    enforce_ontology=True,   # match the CLI: promote entity_type to real labels
)
print(summary.status, summary.texts_considered, summary.texts_deduped)
```

Label promotion is opt-in and mirrors unstructured2graph's flags: the default
(`enforce_ontology=False, promote_labels=False`) leaves entities under the
LightRAG workspace label with an `entity_type` property only; `enforce_ontology=True`
restricts promotion to an ontology (pass `ontology_path=` for a custom one);
`promote_labels=True` promotes every `entity_type` with no vocabulary. See
[unstructured2graph § entity typing](../../unstructured2graph/README.md#entity-typing--ontology).

Extracted entities land in the same LightRAG workspace as any documents
ingested via unstructured2graph by default, so a person or concept mentioned
both in a session and in an ingested document merges into one node. Pass
`entity_workspace=` explicitly to `reconcile_session()` to isolate them instead.

Content is deduplicated by hash before ever reaching the LLM, so re-running a
sweep over already-processed content never re-bills it. Each reconcilable unit
(a message, tool call, tool result, or memory) is truncated to
`MAX_RECONCILABLE_CHARS` (8000) before extraction, but a chatty session still
has many units, so the first run can be substantial. Consider this before
enabling `auto_reconcile` broadly.

## API reference

| Method | Description |
|---|---|
| `setup()` | Create constraints, text index, and reconciliation indexes. Run once on first use. |
| `drop()` | Remove all Memory-related constraints and indexes. |
| `save_memory(user_id, content, *, session_id, memory_id)` | Persist a new Memory. Returns the stored `Memory` object. |
| `get_memories(user_id)` | Return all Memories for a user, newest first. |
| `get_memories_for_session(session_id)` | Return all Memories produced by a session, newest first. |
| `search_memories(user_id, query, *, limit=10)` | Full-text search over Memory content. |
| `update_memory(memory_id, content)` | Replace the content of an existing Memory. Returns `None` if not found. |
| `delete_memory(memory_id)` | Remove a Memory and all its relationships. |
| `async reconcile_session(session_id, *, lightrag_wrapper, actions_graph=None, entity_workspace=None, promote_labels=False, enforce_ontology=False, ontology_path=None)` | Run session reconciliation for one session. `promote_labels`/`enforce_ontology`/`ontology_path` control entity-type label promotion (see above). Returns a `ReconciliationSummary`. Requires the `reconciliation` extra. |
| `get_pending_reconciliation_sessions(*, limit=100)` | Return session IDs marked `reconciliation_status = 'pending'`. |
