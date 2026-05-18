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

## Quick start

```python
from sessions_graph import SessionsGraph

graph = SessionsGraph()   # connects via MEMGRAPH_HOST / MEMGRAPH_PORT env vars
graph.setup()           # creates constraints and the text index (run once)

# Write a memory
mem = graph.save_memory(
    user_id="alice",
    content="Prefers Python over TypeScript",
    session_id="s-abc123",   # optional — links memory to a session for provenance
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
    └─[:HAS_MEMORY]─▶ (:Memory {memory_id, user_id, content, created_at})
                              ▲
              [:PRODUCED_MEMORY]
                              │
                      (:Session {session_id})   ← shared idempotent coordination point
```

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
| `setup()` | Create constraints and text index. Run once on first use. |
| `drop()` | Remove all Memory-related constraints and indexes. |
| `save_memory(user_id, content, *, session_id, memory_id)` | Persist a new Memory. Returns the stored `Memory` object. |
| `get_memories(user_id)` | Return all Memories for a user, newest first. |
| `search_memories(user_id, query, *, limit=10)` | Full-text search over Memory content. |
| `update_memory(memory_id, content)` | Replace the content of an existing Memory. Returns `None` if not found. |
| `delete_memory(memory_id)` | Remove a Memory and all its relationships. |
