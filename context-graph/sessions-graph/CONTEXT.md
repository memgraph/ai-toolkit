# Sessions Graph

Cross-session memory recall. Stores explicit Memories, links them to source sessions, exposes via Python API. Implements design once called **Memory Graph**; named `sessions-graph` since `memory-graph` was taken on PyPI.

## Language

**Sessions Graph**:
Graph component persisting + recalling **Memories** across sessions. Domain = memory, not session management.
_Avoid_: session tracker, session store, session owner

**Memory**:
Free-form text assertion about a user/their work, an agent deliberately saves for later sessions.
_Avoid_: Fact, note, log entry, preference, session record, extracted insight, typed fact

**Memory Write**:
Persisting a Memory via Sessions Graph Python API. MCP exposure deferred until write contract is clear.
_Avoid_: Memory extraction, LLM distillation, passive capture, tool-intercepted write

**Memory Recall**:
Retrieving Memories from Sessions Graph via Python API (later, maybe MCP tool).
_Avoid_: Memory injection, automatic context loading, passive recall

**Memory Owner**:
User identity (`user_id` from `SessionStartEvent`) a Memory belongs to. Every Memory user-owned.
_Avoid_: Memory scope, memory namespace, memory author

**Memory Provenance**:
Session a Memory originated from — recorded as `(:Session)-[:PRODUCED_MEMORY]->(:Memory)`.
_Avoid_: Memory tag, memory scope, memory metadata

**Memory Search**:
Full-text search over Memory nodes' `content` property, via Memgraph full-text index. Primary v1 recall mechanism.
_Avoid_: Semantic search, vector search, embedding retrieval (later phase)

**Session Node**:
Shared `(:Session {session_id})` node joining data from Context Graph components. Any component may `MERGE` it; none owns it.
_Avoid_: session owner, session authority, session registry

**Session Reconciliation**:
Out-of-band, LLM-backed batch process. Dedupes Reconcilable Content, combines into one document per session, uses unstructured2graph to create Chunks + extract entities. Per-session batching lets extraction see facts/references spanning turns ([#297](https://github.com/memgraph/ai-toolkit/issues/297)). Same run creates session's **Episode**. Never creates Memory nodes — Memories are explicit writes.
_Avoid_: Memory extraction, memory reconciliation, auto-memory

**Episode**:
Summary of what happened in one session. Session Reconciliation creates it via separate LLM call over same content used for entity extraction: `(:Session)-[:HAS_EPISODE]->(:Episode {summary, summarized_at})`. Max one per session; re-reconciliation updates it.
_Avoid_: Session Summary (superseded — Episode is a real node now, not a Session property)

**Reconcilable Content**:
Text from a session's Message, ToolCall, ToolResult, Memory content useful for entity extraction. Excludes non-text actions: errors, structured output, permission requests, rate limits.
_Avoid_: Session content (broader — includes ineligible non-text/structured actions)

**Chunk**:
Persisted unit of Reconcilable Content, used for entity extraction. unstructured2graph owns schema + hashing. Sessions Graph links each source Action/Memory to resulting Chunks via `HAS_CHUNK`.
_Avoid_: Memory (Chunk = derived output, not explicit assertion)

**Reconciliation Status**:
Session's `reconciliation_status`: `pending`, `completed`, or `failed`.
_Avoid_: Status, session status (`status` already tracks session completion)

## Relationships

- **Sessions Graph** belongs to broader **Context Graph** family.
- **Memory Write** + **Memory Recall** happen via Sessions Graph Python API, not **Event Protocol**.
- `SessionsGraphConnector` consumes session start/end events. `MERGE`s User + Session nodes, exposes `active_user_id`/`active_session_id`.
- **Memory Owner** connected to their sessions: `(:User)-[:HAD_SESSION]->(:Session)`. Created atomically with User/Session MERGEs on `SESSION_START`.
- Every **Memory** owned by a **Memory Owner** via `HAS_MEMORY`: `(:User)-[:HAS_MEMORY]->(:Memory)`.
- **Memory Provenance** encoded as `(:Session)-[:PRODUCED_MEMORY]->(:Memory)`.
- Scope derived from graph topology, not stored as attribute.
- Session Reconciliation reads Message/ToolCall/ToolResult Actions + Memories, passes text to unstructured2graph.
- Each source Action/Memory links to resulting Chunks: `(:Action|:Memory)-[:HAS_CHUNK]->(:Chunk)`. If combined session document splits into several Chunks, every source links to every Chunk — provenance then session-level, not exact per source.
- Session Reconciliation `MERGE`s `Session-[:HAS_EPISODE]->Episode`: later run updates the Episode, never adds another.
- On `SESSION_END`, `SessionsGraphConnector` only marks reconciliation `pending`. Manual/scheduled CLI sweep, or opt-in background process, runs the LLM work later. Hook subprocesses never run reconciliation.

## Flagged ambiguities

- "scope" can imply stored attribute. Resolved: scope derived from graph topology (ownership + provenance relationships), not a Memory-node attribute.
- "extract" implies passive/LLM-driven capture. Resolved: Memories always written explicitly via Python API — use **Memory Write**.
- "global memory" ambiguous. Resolved: deferred — v1, every Memory user-owned. Cross-user/repo-shared memories out of scope.
- "sessions-graph owns sessions" surfaced during PyPI naming discussion. Resolved: Session nodes = shared idempotent coordination point; none owns them.
- "status" ambiguous: `status` tracks agent session; `reconciliation_status` tracks reconciliation. Never use bare `status` for reconciliation.
- Reconciliation doesn't extend a Memory. Derives Chunks, entities, Episode from Action/Memory content. Use **Session Reconciliation** — never "memory reconciliation"/"memory extraction."
- Episodic memory = `Episode` node linked via `HAS_EPISODE`, not a Session property ([#261](https://github.com/memgraph/ai-toolkit/issues/261)).
- Should Sessions Graph define an Entity? Resolved: no. unstructured2graph owns Entity/Chunk semantics; Sessions Graph only links sources to Chunks.
- `HAS_CHUNK` exact when session produces one Chunk. Several Chunks -> every source links to every Chunk, provenance session-level not exact per source.
