# Sessions Graph

Cross-session memory recall. Stores explicit Memories, links them to source sessions, exposes explicit Memory APIs and hybrid session recall through Python and MCP. Implements design once called **Memory Graph**; named `sessions-graph` since `memory-graph` was taken on PyPI.

## Language

**Sessions Graph**:
Graph component owning User and Session lifecycle, explicit **Memories**, reconciliation, and hybrid session recall.
_Avoid_: treating session tracking as the whole memory component

**Memory**:
Free-form text assertion about a user/their work, an agent deliberately saves for later sessions. Stored as one file in the user's `/memories` tree: `path` unique per user.
_Avoid_: Fact, note, log entry, preference, session record, extracted insight, typed fact

**Memory File**:
A Memory seen through the memory tool: a path under `/memories` plus its text. Directories aren't stored — they are shared path prefixes. `/memories/` = applies everywhere; `/memories/projects/<key>/` = one project.
_Avoid_: Document, note file, memory folder (as a stored thing)

**Memory Backend**:
Which memory a harness's model uses: its own built-in memory (`native`) or Context Graph (`context-graph`). Per user, opt-in (`[memory] backend` in the config file). Never both at once.
_Avoid_: Memory mode, memory provider

**Memory Write**:
Persisting a Memory via Sessions Graph Python API, or by the model through the `memory` tool (memory tool commands, ADR 0006 in agent-context-graph). The only write a model makes over MCP; everything derived stays background-only.
_Avoid_: Memory extraction, LLM distillation, passive capture, tool-intercepted write

**Memory Recall**:
Retrieving explicit Memories through the Python API. The separate `recall` MCP tool returns source turns and typed facts from the user's sessions.
_Avoid_: Memory injection, automatic context loading, passive recall

**Memory Owner**:
User identity (`user_id` from `SessionStartEvent`) a Memory belongs to. Every Memory user-owned.
_Avoid_: Memory scope, memory namespace, memory author

**Memory Provenance**:
Session a Memory originated from — recorded as `(:Session)-[:PRODUCED_MEMORY]->(:Memory)`.
_Avoid_: Memory tag, memory scope, memory metadata

**Memory Search**:
Full-text search over Memory nodes' `content` property, via Memgraph full-text index. Mechanism for explicit Memory search; hybrid session recall also uses vector search.
_Avoid_: conflating explicit Memory search with hybrid session recall

**Session Node**:
Shared `(:Session {session_id})` node joining data from Context Graph components. Any component may `MERGE` it; sessions-graph owns its lifecycle and the User/HAD_SESSION relationship.
_Avoid_: treating other components' MERGEs as lifecycle ownership

**Session Reconciliation**:
Out-of-band batch process using local GLiNER2 entity extraction and an LLM summary. Dedupes Reconcilable Content, combines into one document per session, uses unstructured2graph to create Chunks and extract entities using the adopted HyGM model (or the default model). Per-session batching lets extraction see facts/references spanning turns ([#297](https://github.com/memgraph/ai-toolkit/issues/297)). Same run creates session's **Episode**. Never creates Memory nodes — Memories are explicit writes. Doesn't read Memories either: each Memory File has its own **Memory File Reconciliation**.
_Avoid_: Memory extraction, memory reconciliation, auto-memory

**Memory File Reconciliation**:
Per-file counterpart of Session Reconciliation (`reconcile_memory`). A write marks the file `extraction_status = 'pending'`; the `reconcile --pending` sweep removes what the file's previous text produced, then extracts Chunks and entities from its current text. No LLM, no Episode. Per file, not per session, because a file outlives the session that first wrote it and changes across many. `memory` tool calls are excluded from session content, so the text isn't extracted twice.
_Avoid_: Memory extraction (implies deriving Memories), memory reconciliation (ambiguous with Session Reconciliation)

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
- `SessionsGraphConnector` consumes session start/end and turn-end events. `MERGE`s User + Session nodes, exposes `active_user_id`/`active_session_id`.
- **Memory Owner** connected to their sessions: `(:User)-[:HAD_SESSION]->(:Session)`. Created atomically with User/Session MERGEs on `SESSION_START`.
- Every **Memory** owned by a **Memory Owner** via `HAS_MEMORY`: `(:User)-[:HAS_MEMORY]->(:Memory)`.
- A **Memory File** under `/memories/projects/<key>/` is `(:Memory)-[:ABOUT]->(:Project {key})`; the key is one path segment.
- **Memory Provenance** encoded as `(:Session)-[:PRODUCED_MEMORY]->(:Memory)`.
- Scope derived from graph topology, not stored as attribute.
- Session Reconciliation reads Message/ToolCall/ToolResult Actions + Memories, passes text to unstructured2graph.
- Each source Action/Memory links to resulting Chunks: `(:Action|:Memory)-[:HAS_CHUNK]->(:Chunk)`. If combined session document splits into several Chunks, every source links to every Chunk — these links express session-level provenance. `MENTIONED_IN.sources` and typed edges' `source_id` retain exact turn provenance used by recall.
- Session Reconciliation `MERGE`s `Session-[:HAS_EPISODE]->Episode`: later run updates the Episode, never adds another.
- On `SESSION_END`, `SessionsGraphConnector` only marks reconciliation `pending`. Manual/scheduled CLI sweep, or opt-in background process, runs the LLM work later. Hook subprocesses never run reconciliation.

## Flagged ambiguities

- "scope" can imply stored attribute. Resolved: scope derived from graph topology (ownership + provenance relationships), not a Memory-node attribute.
- "extract" implies passive/LLM-driven capture. Resolved: Memories always written explicitly — Python API or the `memory` tool — use **Memory Write**.
- "MCP is read-only" (#259) vs the `memory` tool. Resolved (ADR 0006, map #484): MCP may write `(:Memory)` files only; chunks, entities, Episodes, Procedures stay background-only.
- "global memory" ambiguous. Resolved: deferred — v1, every Memory user-owned. Cross-user/repo-shared memories out of scope.
- "sessions-graph owns sessions" surfaced during PyPI naming discussion. Resolved: Session nodes = shared idempotent coordination point; sessions-graph owns lifecycle while all components may MERGE them.
- "status" ambiguous: `status` tracks agent session; `reconciliation_status` tracks reconciliation. Never use bare `status` for reconciliation.
- Reconciliation doesn't extend a Memory. Session Reconciliation derives Chunks, entities, Episode from Action content; **Memory File Reconciliation** derives Chunks and entities from one Memory File. Never "memory reconciliation"/"memory extraction."
- Episodic memory = `Episode` node linked via `HAS_EPISODE`, not a Session property ([#261](https://github.com/memgraph/ai-toolkit/issues/261)).
- Should Sessions Graph define an Entity? Resolved: no. unstructured2graph owns Entity/Chunk semantics; Sessions Graph only links sources to Chunks.
- `HAS_CHUNK` exact when session produces one Chunk. Several Chunks -> every source links to every Chunk. Recall narrows this using `MENTIONED_IN.sources` and typed edges' `source_id`.
