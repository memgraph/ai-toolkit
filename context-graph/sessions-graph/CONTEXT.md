# Sessions Graph

Sessions Graph provides cross-session memory recall. It stores explicit Memories,
links them to their source sessions, and exposes them through a Python API. The
package implements the design once called **Memory Graph**; `sessions-graph` was
chosen because `memory-graph` was unavailable on PyPI.

## Language

**Sessions Graph**:
The graph component that persists and recalls **Memories** across sessions. Its
domain is memory, not session management.
_Avoid_: session tracker, session store, session owner

**Memory**:
A free-form text assertion about a user or their work that an agent deliberately
saves for later sessions.
_Avoid_: Fact, note, log entry, preference, session record, extracted insight, typed fact

**Memory Write**:
Persisting a Memory through the Sessions Graph Python API. MCP exposure is
deferred until the write contract is clear.
_Avoid_: Memory extraction, LLM distillation, passive capture, tool-intercepted write

**Memory Recall**:
The act of retrieving Memories from Sessions Graph via the Python API (or later an MCP tool).
_Avoid_: Memory injection, automatic context loading, passive recall

**Memory Owner**:
The user identity (`user_id` from `SessionStartEvent`) that a Memory belongs to. Every Memory is user-owned.
_Avoid_: Memory scope, memory namespace, memory author

**Memory Provenance**:
The session from which a Memory originated — recorded as a `(:Session)-[:PRODUCED_MEMORY]->(:Memory)` relationship.
_Avoid_: Memory tag, memory scope, memory metadata

**Memory Search**:
Full-text search over the `content` property of Memory nodes, backed by a Memgraph full-text index. The primary recall mechanism for v1.
_Avoid_: Semantic search, vector search, embedding retrieval (deferred to a later phase)

**Session Node**:
A shared `(:Session {session_id})` node that joins data from Context Graph
components. Any component may `MERGE` it. No component owns it.
_Avoid_: session owner, session authority, session registry

**Session Reconciliation**:
An out-of-band, LLM-backed batch process. It deduplicates Reconcilable Content,
combines it into one document per session, then uses unstructured2graph to create
Chunks and extract entities. Per-session batching lets extraction see facts and
references that span turns ([#297](https://github.com/memgraph/ai-toolkit/issues/297)).
The same run also creates the session's **Episode**. It never creates Memory
nodes; Memories are explicit writes.
_Avoid_: Memory extraction, memory reconciliation, auto-memory

**Episode**:
A summary of what happened in one session. Session Reconciliation creates it with
a separate LLM call over the same content used for entity extraction:
`(:Session)-[:HAS_EPISODE]->(:Episode {summary, summarized_at})`. A session has at
most one Episode; re-reconciliation updates it.
_Avoid_: Session Summary (superseded — an Episode is a real node now, not a property on Session)

**Reconcilable Content**:
Text from a session's Message, ToolCall, ToolResult, and Memory content that is
useful for entity extraction. Non-text actions such as errors, structured output,
permission requests, and rate limits are excluded.
_Avoid_: Session content (broader — includes non-text/structured actions that are not eligible)

**Chunk**:
A persisted unit of Reconcilable Content used for entity extraction.
unstructured2graph owns its schema and hashing. Sessions Graph links each source
Action or Memory to resulting Chunks with `HAS_CHUNK`.
_Avoid_: Memory (a Chunk is derived output, not an explicit assertion)

**Reconciliation Status**:
The Session's `reconciliation_status`: `pending`, `completed`, or `failed`.
_Avoid_: Status, session status (`status` already tracks session completion)

## Relationships

- **Sessions Graph** belongs to the broader **Context Graph** family.
- A **Memory Write** and **Memory Recall** happen through the Sessions Graph Python API, not through the **Event Protocol**.
- `SessionsGraphConnector` consumes session start/end events. It `MERGE`s User and
  Session nodes and exposes `active_user_id` and `active_session_id`.
- A **Memory Owner** is connected to their sessions: `(:User)-[:HAD_SESSION]->(:Session)`. This edge is created atomically with the User and Session MERGEs on `SESSION_START`.
- Every **Memory** is owned by a **Memory Owner** via a `HAS_MEMORY` relationship: `(:User)-[:HAS_MEMORY]->(:Memory)`.
- **Memory Provenance** is encoded as: `(:Session)-[:PRODUCED_MEMORY]->(:Memory)`.
- Scope is derived from graph topology, not stored as an attribute.
- Session Reconciliation reads Message, ToolCall, and ToolResult Actions plus
  Memories, then passes their text to unstructured2graph.
- Each source Action or Memory links to resulting Chunks:
  `(:Action|:Memory)-[:HAS_CHUNK]->(:Chunk)`. If the combined session document
  splits into several Chunks, every source links to every Chunk. Provenance is
  then session-level, not exact per source.
- Session Reconciliation `MERGE`s `Session-[:HAS_EPISODE]->Episode`, so a later
  run updates the Episode instead of adding another.
- On `SESSION_END`, `SessionsGraphConnector` only marks reconciliation `pending`.
  A manual or scheduled CLI sweep, or an opt-in background process, runs the LLM
  work later. Hook subprocesses never run reconciliation.

## Flagged ambiguities

- "scope" can imply a stored attribute. Resolved: scope is derived from graph topology (ownership and provenance relationships), not from an attribute on the Memory node.
- "extract" implies passive or LLM-driven capture. Resolved: Memories are always written explicitly via the Python API; use **Memory Write**.
- "global memory" is ambiguous. Resolved: deferred — for v1, every Memory is user-owned. Cross-user or repo-shared memories are out of scope.
- "sessions-graph owns sessions" was introduced during the PyPI naming discussion. Resolved: Session nodes are a shared idempotent coordination point; no component owns them.
- "status" is ambiguous. `status` tracks the agent session;
  `reconciliation_status` tracks reconciliation. Never use bare `status` for
  reconciliation.
- Reconciliation does not extend a Memory. It derives Chunks, entities, and an
  Episode from Action and Memory content. Use **Session Reconciliation**, never
  "memory reconciliation" or "memory extraction."
- Episodic memory is an `Episode` node linked by `HAS_EPISODE`, not a Session
  property ([#261](https://github.com/memgraph/ai-toolkit/issues/261)).
- Whether Sessions Graph should define an Entity. Resolved: no. unstructured2graph
  owns Entity and Chunk semantics; Sessions Graph only links sources to Chunks.
- `HAS_CHUNK` is exact when a session produces one Chunk. With several Chunks,
  every source links to every Chunk, so provenance is session-level rather than
  exact per source.
