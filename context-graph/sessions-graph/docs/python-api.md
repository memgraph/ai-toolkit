# Memory files and Python API

[Back to Sessions Graph](../README.md)

## Memory files

`memory_store(user_id)` exposes a user's Memories as the file tree the Claude
API memory tool (`memory_20250818`) works on: `view`, `create`, `str_replace`,
`insert`, `delete` and `rename` on paths under `/memories`, returning the
result and error strings that tool documents. Directories aren't stored; they
are the path prefixes their files share.

```python
store = graph.memory_store("alice", session_id="s-1")
store.create(
    "/memories/feedback/testing.md", "---\ndescription: Prefer real Memgraph in tests\ntype: feedback\n---\n..."
)
store.view("/memories")  # listing, two levels deep
store.execute({"command": "view", "path": "/memories/feedback/testing.md"})  # a tool call's input as-is
```

- **Paths.** Must start with `/memories`. `.`/`..` segments, backslashes,
  control characters and percent-encoded dots or slashes are rejected, and
  repeated slashes collapse.
- **Writes.** `create` overwrites an existing file. An edit
  (`str_replace`/`insert`) applies only if the file still holds what the edit
  was computed from, so a stale read from another session fails instead of
  clobbering. A file can't be empty or larger than 100 KB.
- **Views.** A file view stops near 16,000 characters and says how to page on
  with `view_range`.
- **History.** Every write that discards text keeps it first as a
  `(:MemoryVersion)`, created in the same query as the write. That covers an
  overwriting `create`, `str_replace`, `insert`, and `delete` (marked
  `deleted`). The newest 20 per file are kept. `store.versions(path)` lists
  them, newest first; a live file's history follows it across renames, and a
  deleted file's history stays under its last path.
- **Provenance and projects.** With `session_id`, every write records
  `(:Session)-[:PRODUCED_MEMORY]->(:Memory)`. A file under
  `/memories/projects/<key>/` is linked `-[:ABOUT]->(:Project {key})`, and a
  rename re-links it.

Apps built on the Claude API get the same files through the SDK's tool
runner (`pip install 'sessions-graph[anthropic]'`). The application supplies
the user; the model never does.

```python
from sessions_graph.anthropic_memory import MemgraphMemoryTool  # AsyncMemgraphMemoryTool for the async client

memory = MemgraphMemoryTool(graph, user_id="alice", session_id="request-42")
client.beta.messages.tool_runner(model=..., max_tokens=..., tools=[memory], messages=[...]).until_done()
```

Harnesses reach it as the `memory` tool, which memgraph-context-graph serves once
the user makes Context Graph their memory backend; see
[memgraph-context-graph § Memory](../../memgraph-context-graph/README.md#memory-context-graph-as-the-harnesss-memory).

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

## API reference

| Method | Description |
|---|---|
| `setup()` | Create constraints, text index, and reconciliation indexes. Run once on first use. |
| `save_memory(user_id, content, *, session_id, memory_id, path)` | Persist a new Memory at `path` (default `/memories/notes/<memory_id>.md`). Returns the stored `Memory` object. |
| `memory_store(user_id, *, session_id=None)` | The memory tool's file commands over the user's Memories. See [Memory files](../README.md#memory-files). |
| `get_memories(user_id)` | Return all Memories for a user, newest first. |
| `get_memories_for_session(session_id)` | Return all Memories produced by a session, newest first. |
| `search_memories(user_id, query, *, limit=10)` | Full-text search over Memory content. |
| `update_memory(memory_id, content)` | Replace the content of an existing Memory. Returns `None` if not found. |
| `delete_memory(memory_id)` | Remove a Memory and all its relationships. |
| `recall(user_id, question, *, config=None, model=DEFAULT_EMBEDDING_MODEL)` | What the user's own sessions hold about `question`, as `Recalled` (turns and facts; `lines()`, `render(today)`, `to_json()`). See [Recall](../README.md#recall). |
| `record_session_start(session_id, *, user_id, project_key)` | MERGE the Session, owned by `user_id` and in `project_key` when known. The connector calls it on session start. |
| `reconciler` | The graph's `Reconciler`, below: extraction and embedding over the same connection. |

`graph.reconciler` (`sessions_graph.reconciler.Reconciler`) turns sessions and
memory files into entities and vectors, and is the one writer of their
processing status (`reconciliation_*`, `reextract*`, `embedding_*`,
`extraction_status`):

| Method | Description |
|---|---|
| `async reconcile(session_id, *, extraction_backend=None, promote_labels=False, enforce_ontology=False, ontology_path=None, embedding_model=DEFAULT_EMBEDDING_MODEL, reextract=False)` | Reconcile one session, then embed what it wrote (see [Embeddings for recall](../README.md#embeddings-for-recall)). Extracts with GLiNER2 under the user's adopted ontology version unless `extraction_backend` overrides it. No LLM. Returns a `ReconciliationSummary`. Requires the `reconciliation` extra. |
| `async reconcile_batch(session_ids, *, lightrag_wrapper, ...)` | Reconcile many sessions with LightRAG as one processing pass. |
| `async reconcile_memory(memory_id, *, extraction_backend=None, enforce_ontology=True, embedding_model=DEFAULT_EMBEDDING_MODEL)` | Re-extract one memory file, replacing what its previous text produced. See [Session reconciliation](../README.md#session-reconciliation). |
| `mark_pending(session_id)` | Queue the session for the next `reconcile --pending` sweep. |
| `pending_sessions(*, limit=100)` | Session IDs marked `reconciliation_status = 'pending'`. |
| `pending_memories(*, limit=100)` | Memory IDs whose file changed since it was last reconciled. |
| `embed(session_id, *, model=DEFAULT_EMBEDDING_MODEL)` | Embed the session's messages, entities and edges that lack a vector from `model`, inside Memgraph. Returns counts as `Embedded`; records the outcome on the Session. Raises `EmbeddingUnavailableError` without MAGE or when the model can't load. |
| `pending_embeddings(*, model=DEFAULT_EMBEDDING_MODEL, limit=100)` | Session IDs whose embedding failed, never ran, or used another model. |
| `session_document(session_id, user_id)` / `session_user(session_id)` / `backend_for(model)` | What derivation reads a session as, the session's user, and the cached GLiNER2 backend for an ontology model. |

## Integration with Memgraph Context Graph

An app built on the Claude Agent SDK or the OpenAI Agents SDK gets this package's memory — capture, the session-start memory index, and the `recall`/`memory` tools — from `memgraph_context_graph.sdk`; see [Memory for SDK apps](../../memgraph-context-graph/README.md#memory-for-sdk-apps).

To wire the connector yourself, add `SessionsGraphConnector` to an `AgentLink`. It tracks the active `session_id` and `user_id` from `SessionStartEvent`, so you can reference them when saving memories.

```python
from sessions_graph import SessionsGraph
from sessions_graph.connector import SessionsGraphConnector
from memgraph_context_graph import AgentLink
from memgraph_context_graph.events import SessionStartEvent

graph = SessionsGraph()
graph.setup()

connector = SessionsGraphConnector(graph)
link = AgentLink()
link.add_connector(connector)

link.emit(SessionStartEvent(session_id="s-abc123", user_id="alice"))

# During the session, save memories via the Python API:
graph.save_memory(
    user_id=connector.active_user_id,
    content="User works primarily in the ai-toolkit repository",
    session_id=connector.active_session_id,
)
```
