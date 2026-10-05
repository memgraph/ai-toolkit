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
through [unstructured2graph](../../unstructured2graph/)'s chunk + entity-extraction
pipeline -- LightRAG by default, overridable to another `ExtractionBackend`
(e.g. GLiNER2) via `reconcile_session(..., extraction_backend=...)` -- turning
it into queryable graph entities linked
back to the session that produced them — see
[`CONTEXT.md`](./CONTEXT.md#language) for the **Session Reconciliation** /
**Reconcilable Content** / **Reconciliation Status** terminology.

The same pass also writes the session's **episodic memory**: an
`(:Episode {summary, summarized_at})` node linked via
`(:Session)-[:HAS_EPISODE]->(:Episode)` (at most one per session — re-running
reconciliation updates it rather than adding another), produced by a second,
dedicated LLM call over the same deduped session text — a "what happened in
this session" gist, not the structured entity graph. This is what a "what did
we do last time?" recall query actually reads.

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
  session ends. For hook-based runtimes (Claude Code, Codex), set this
  persistently:

  ```bash
  agent-context-graph config set reconcile.auto_reconcile true
  ```

  This writes `[reconcile] auto_reconcile = true` to
  `~/.config/context-graph/config.toml`, which
  `agent_context_graph.hooks.runner._add_sessions_graph_connector` reads on
  every hook invocation via `resolve_auto_reconcile()` — no shell/session
  restart quirks, since hook subprocesses don't reliably inherit shell
  profile environment variables. Direct SDK integrations that construct
  `SessionsGraphConnector` themselves must instead pass
  `SessionsGraphConnector(graph, auto_reconcile=True)` explicitly — per ADR
  0002 (config-file-only-hook-resolution), no ambient environment variable
  is consulted here. This is fire-and-forget — if the process dies before
  finishing (machine sleep, crash), the session stays `pending` and
  `sessions-graph reconcile --pending` is the reliable backfill.

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
    enforce_ontology=True,  # match the CLI: promote entity_type to real labels
)
print(summary.status, summary.texts_considered, summary.texts_deduped, summary.summary_written)
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

## Recall

`recall(user_id, question)` returns what one user's own past sessions hold
about a question: the evidence, not an answer. The caller's model answers
from it.

```python
recalled = graph.recall("alice", "Which database did we pick for the cache?")
print(recalled.render(today="2026-10-05"))  # header with reading rules, then the rows
recalled.to_json()  # the same turns and facts as data
```

Five lanes, each over the user's own history only:

| Lane | Finds |
|---|---|
| `turns` | messages nearest the question by vector |
| `text` | messages matching it by full-text search |
| `entities` | entities nearest it by vector, and each one's facts |
| `facts` | extracted facts nearest it by vector |
| `user_facts` | the user's facts of the relation types nearest it, across all sessions |

Then the turns the facts were read from are added. Turns come first, then
facts, each group in time order. This is the hybrid retrieval
`context-graph-eval run --retrieval-strategy hybrid` benchmarks, through
this same code: 89/100 on LongMemEval's own judge (run `g417-recall-r1`,
100 questions, map #390).

Widths default to the benchmarked setup; `RecallConfig.from_mapping(...)`
overrides them (e.g. from a config file's `[recall]` section). Similarity is
computed exactly over the user's own vectors, which costs time linear in
their history: about 0.6 s for a few hundred messages, 0.7 s at 10k, 6.7 s at
100k. Without MAGE, recall runs text search alone and says so in its result.

## Embeddings for recall

Recall searches a user's history by vector as well as by text, so three
units get an `embedding` property, computed **inside Memgraph** by MAGE's
`embeddings` module (no model runs on the host):

| Unit | Text embedded |
|---|---|
| user and assistant messages (`:Action`) | `text`, the plain message text Actions Graph writes |
| entities | their `text`, for every entity mentioned in one of the session's chunks |
| extracted edges | `"<head> <type> <tail>. <r.text>"`: the fact and the sentence it was read from |

Each vector also records its `embedding_model`. A vector from a different
model counts as missing and is replaced, so vectors from two models never
mix. The default model is `BAAI/bge-small-en-v1.5` (384 dimensions); change
it with:

```bash
agent-context-graph config set recall.embedding_model <huggingface-model-name>
```

When it runs:

- **Session end:** `SessionsGraphConnector` always spawns a detached
  `sessions-graph embed --session <id>`, whether or not `auto_reconcile` is
  on. Embedding needs no LLM, and the hook never waits on the model.
- **Reconciliation:** after extraction, the session's new entities and edges
  are embedded. An embedding failure never fails the reconciliation.
- **Catch-up:** `sessions-graph embed --pending` embeds every session whose
  embedding failed, never ran, or used another model.

The outcome is on the Session: `embedding_status` is `completed` (with
`embedding_model`) or `failed` (with `embedding_error`).

This needs Memgraph with MAGE (`memgraph/memgraph-mage`), with 2 GiB of
memory or more for the default model. The model downloads inside Memgraph on
first use. On plain `memgraph/memgraph` every session records `failed`, and
`agent-context-graph doctor --connector sessions-graph` reports the
`embeddings` check as failing.

```bash
sessions-graph embed --session s-abc123
sessions-graph embed --pending --limit 50 --model BAAI/bge-small-en-v1.5
```

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
| `async reconcile_session(session_id, *, lightrag_wrapper, extraction_backend=None, actions_graph=None, entity_workspace=None, promote_labels=False, enforce_ontology=False, ontology_path=None, embedding_model=DEFAULT_EMBEDDING_MODEL)` | Run session reconciliation for one session, then embed what it wrote (see [Embeddings for recall](#embeddings-for-recall)). `extraction_backend` overrides entity extraction to another `ExtractionBackend` (e.g. GLiNER2); `lightrag_wrapper` is always required regardless, since the narrative summary is always produced via its LLM. `promote_labels`/`enforce_ontology`/`ontology_path` control entity-type label promotion (see above). Returns a `ReconciliationSummary`. Requires the `reconciliation` extra. |
| `get_pending_reconciliation_sessions(*, limit=100)` | Return session IDs marked `reconciliation_status = 'pending'`. |
| `recall(user_id, question, *, config=None, model=DEFAULT_EMBEDDING_MODEL)` | What the user's own sessions hold about `question`, as `Recalled` (turns and facts; `lines()`, `render(today)`, `to_json()`). See [Recall](#recall). |
| `embed_session(session_id, *, model=DEFAULT_EMBEDDING_MODEL)` | Embed the session's messages, entities and edges that lack a vector from `model`, inside Memgraph. Returns counts as `Embedded`; records the outcome on the Session. Raises `EmbeddingUnavailableError` without MAGE or when the model can't load. |
| `get_pending_embedding_sessions(*, model=DEFAULT_EMBEDDING_MODEL, limit=100)` | Session IDs whose embedding failed, never ran, or used another model. |
