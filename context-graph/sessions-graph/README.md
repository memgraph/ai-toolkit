# Sessions Graph

Sessions Graph is the [Context Graph](../CONTEXT-MAP.md) component for session context and cross-session recall. It owns `(:User)` and `(:User)-[:HAD_SESSION]->(:Session)`; the `(:Session)` node itself is shared by every component. It reconciles sessions into extracted entities, embeds them, and answers [recall](#recall) over a user's history. It also stores free-form text assertions — **Memories** — that agents write explicitly.

Requires **Memgraph ≥ 3.6** (text search is stable from that release).

## Installation

```bash
pip install sessions-graph
```

To use with Memgraph Context Graph:

```bash
pip install sessions-graph[memgraph-context-graph]
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

## Integration with Memgraph Context Graph

An app built on the Claude Agent SDK or the OpenAI Agents SDK gets this package's memory — capture, the session-start memory index, and the `recall`/`memory` tools — from `memgraph_context_graph.sdk`; see [Memory for SDK apps](../memgraph-context-graph/README.md#memory-for-sdk-apps).

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

## Graph schema

```
(:User {user_id})
    ├─[:HAS_MEMORY]──▶ (:Memory {memory_id, user_id, path, content, created_at, updated_at})
    │                          │
    │                       [:ABOUT]──▶ (:Project {key})   (files under /memories/projects/<key>/)
    │                          ▲                        │
    │            [:PRODUCED_MEMORY]              [:HAS_CHUNK]
    │                          │                        ▼
    └─[:HAD_SESSION]─▶ (:Session {session_id,   (:Chunk {hash, text})
                                  project_key,               ▲
                                  reconciliation_status,     │
                                  reconciled_at})  [:HAS_CHUNK]
                              │                        │
                        [:HAS_ACTION]                  │
                              ▼                        │
                        (:Action) ─────────────────────┘
                              │
                                              (:Entity)-[:MENTIONED_IN]->(:Chunk)
```

Every Memory is a file in the user's `/memories` tree: `path` is unique per
user, and a Memory under `/memories/projects/<key>/` links to that project. See
[Memory files](#memory-files).

`(:User)-[:HAD_SESSION]->(:Session)` is written by `SessionsGraphConnector` on
session start; it is the join key other Context Graph components hang off of.
The same write sets `project_key`, the start directory's checkout keyed as
memory folders are, so the next session in that project can open with where the
user left off (`sessions_graph.resume`, shown by the `recall` tool's session-start
text).

`(:Action)` is owned by [Actions Graph](../actions-graph/); `(:Chunk)` and the
extracted entity nodes are owned by
[unstructured2graph](../../unstructured2graph/). See [Session
reconciliation](#session-reconciliation) below for how they get linked.

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
[memgraph-context-graph § Memory](../memgraph-context-graph/README.md#memory-context-graph-as-the-harnesss-memory).

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

Session reconciliation runs a session's Actions Graph content (Messages,
ToolCalls, ToolResults) through [unstructured2graph](../../unstructured2graph/)
as one document, extracting entities with local GLiNER2 under the user's
[ontology version](#ontology-versions) (or another `ExtractionBackend` passed
as `graph.reconciler.reconcile(..., extraction_backend=...)`), and links them back to
the session that produced them. See [`CONTEXT.md`](./CONTEXT.md#language) for
the terminology.

Memory files are reconciled on their own, not with a session, because a file
outlives the session that first wrote it.
- **Trigger.** Every write marks the file `extraction_status = 'pending'`.
- **What runs.** `graph.reconciler.reconcile_memory(memory_id)` first removes what the file's
  previous text produced. That covers relations carrying its id as
  `source_id`, its share of `MENTIONED_IN.sources`, and any chunks and
  entities nothing else backs. It then extracts from the current text, with
  no LLM.
- **Sweep.** `sessions-graph reconcile --pending` processes pending files
  before sessions.
- **Deletes.** Deleting a file removes what it produced the same way.
- **No double extraction.** `memory` tool calls are left out of session
  content, so their text isn't extracted twice.

Reconciliation writes no session summary: recall answers from the turns
themselves and uses the extracted graph as an index into them.

This requires the `sessions-graph[reconciliation]` extra and no LLM API key.
A key (`llm.anthropic_api_key` or `llm.openai_api_key`) only enables
[ontology derivation](#ontology-versions); without one, users stay on their
adopted ontology.

**Extraction settings.** The CLI reads the config file's `[extraction]`
(`model`, `entity_threshold`, `relation_threshold`, `chunk_size`,
`chunk_overlap`) into an `ExtractionConfig`; in code, pass
`SessionsGraph(extraction=ExtractionConfig(...))`. Reconciliation,
derivation's re-extraction and its observer all use it, so a session is
re-read the way it was first read. Unset fields keep GLiNER2's defaults.

**Reconciliation never runs inside a hook itself.** Local GLiNER2 extraction
can exceed the timeout that hook runtimes enforce on hook commands. Instead:

- On `SESSION_END` and on `TURN_END`, `SessionsGraphConnector` cheaply marks
  the session `reconciliation_status = 'pending'` — no LLM calls, safe inside
  the hook. Turn ends matter because several runtimes (Codex, Antigravity
  CLI, `opencode run`) never report a session end; their sessions are only
  ever reconciled through the `--pending` sweep.
- The actual reconciliation run happens out-of-band, via the CLI:

  ```bash
  # Reconcile one session
  sessions-graph reconcile --session s-abc123

  # Sweep every session still marked 'pending' (e.g. from cron)
  sessions-graph reconcile --pending --limit 50
  ```

  The CLI runs with **`enforce_ontology=True`**: extracted entities get real
  type labels (`:Person`, `:Organization`, …) from the user's ontology
  version, and anything outside it is kept but flagged
  `ontology_conformant = false`. See [entity typing](../../unstructured2graph/README.md#entity-typing--ontology).

- Or, if you want it triggered automatically without a manual/cron step, opt
  in to a **best-effort detached background process** spawned right after a
  session ends (never on a turn end). For hook-based runtimes, set this
  persistently:

  ```bash
  memgraph-context-graph config set reconcile.auto_reconcile true
  ```

  This writes `[reconcile] auto_reconcile = true` to the config file, which
  every hook invocation reads (hook subprocesses don't inherit shell profile
  variables, so no environment variable is consulted — ADR 0002). The
  spawned process is handed the config file's path, not its values, and
  reads its connection and keys from that file itself (ADRs 0003, 0007). Direct SDK
  integrations pass `SessionsGraphConnector(graph, auto_reconcile=True)`. It
  is fire-and-forget: if the process dies, the session stays `pending` and
  `sessions-graph reconcile --pending` is the reliable backfill.

Programmatically:

```python
from sessions_graph import SessionsGraph

graph = SessionsGraph()
graph.setup()

summary = await graph.reconciler.reconcile(
    "s-abc123",
    enforce_ontology=True,  # match the CLI: promote entity_type to real labels
)
print(summary.status, summary.texts_considered, summary.texts_deduped)
```

Label promotion mirrors unstructured2graph's flags: by default entities keep
the backend's workspace label (`gliner2`) and an `entity_type` property only;
`enforce_ontology=True` restricts promotion to an ontology (`ontology_path=`
for a custom one); `promote_labels=True` promotes every `entity_type`. See
[unstructured2graph § entity typing](../../unstructured2graph/README.md#entity-typing--ontology).

Extracted entities land in the backend's workspace, shared with documents
ingested through unstructured2graph, so an entity mentioned in both merges
into one node.

Content is deduplicated by hash before extraction, and each reconcilable unit
(a message, tool call, tool result, or memory) is truncated to
`MAX_RECONCILABLE_CHARS` (8000).

## Ontology versions

Reconciliation extracts each session under its user's **adopted ontology
version**, and records which one on the Session as `ontology_version`. A user
with none is on version 0, `hygm.default_model()`: the fixed core (User,
Person and the value types), Organization/Location/Event, and the catch-alls
Topic and Artifact.

Versions are graph nodes: one `(:OntologyVersion {user_id, version, status,
source, derive, model, pinned, source_hash})` each, holding the model as JSON.
`(:User)-[:ADOPTED]->` points at the current one, `NEXT` chains them, and
adopting a version moves `ADOPTED` in one transaction.

A user can supply their own schema (a `ManualStrategy` YAML, see
[hygm](../../hygm/README.md)):

```bash
sessions-graph ontology load --file coding.yaml            # --user defaults to identity.user_id
sessions-graph ontology load --file longmemeval.yaml --derive off
sessions-graph ontology show
```

- `--derive extend` (the default) adds the fixed core to the schema and pins
  its types: derivation may add types beside them, never merge, rename or
  retire one.
- `--derive off` uses the schema exactly as given, and nothing is derived.
  That's how a benchmark stays on a fixed vocabulary.

Or set it in the config file, so it follows edits:

```bash
memgraph-context-graph config set ontology.path ~/schemas/coding.yaml
memgraph-context-graph config set ontology.derive extend
```

Each `sessions-graph reconcile` applies that file to the configured user
whenever its content or `derive` changed since their adopted version came
from it. A file that doesn't validate is reported and the adopted version
kept. Removing the setting deletes nothing.

### Learning the ontology from the user's sessions

`sessions-graph derive` grows a user's model from their own sessions. You don't
normally run it by hand: `sessions-graph reconcile` starts it, detached, when a
user's count of qualifying sessions reaches 2, 4, 8, … 128, and then every 128.
A qualifying session is a reconciled one with at least 2 user turns and 1,000
characters of the user's own words.

```bash
sessions-graph derive                       # --user defaults to identity.user_id
sessions-graph derive --force --seeds 1     # now, without waiting for a milestone
```

One run:

1. Takes an expiring claim on the user, so only one run at a time.
2. Reads the sessions since the last adopted run, sampled down to 64.
3. Derives up to three candidates (`hygm.LlmRecommendationStrategy`, observed
   with local GLiNER2).
4. Gates each candidate against the current model. When the sampled delta
   contains at least 8 sessions, the gate holds out a quarter (at most 8 sessions); smaller
   runs gate in-sample. A candidate passes when it is no worse on catch-all
   share (mentions typed `Topic`/`Artifact`) and on coverage (user turns with a
   typed relation).

The best passing candidate becomes the next version. Merged types are then
relabelled in place, and the run's delta is re-extracted when types were added.
If nothing passes, the candidates are kept as rejected versions and the delta
rolls into the next run.

Every version records its changelog, observation counts, retired-type pool and
the gate's numbers. The LLM is Anthropic when `llm.anthropic_api_key` is set
(default `claude-sonnet-5-5`), else OpenAI (default `gpt-5`); `--model`
overrides it. A version with `derive = off` is never derived from. Without
either key, `reconcile` never spawns `derive` and the adopted version stays.

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
| `user_facts` | every fact of the relation types nearest it, read from the user's turns across all sessions, whatever its head. Types rank by name and description from the user's ontology version, and retired types stay readable |

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

For bounded source context, install `sessions-graph[compact]` and opt in through
`RecallConfig(packing="exchanges", token_budget=6000, supplemental_search=True)`
or the same keys in `[recall]`. This keeps the regular `Recalled` interface,
including `turns`, `memories`, `lines()`, `render()` and `to_json()`. Extracted
facts rank source exchanges but are not rendered as separate rows in this mode.
The row budget includes memory notes, exchange headers and separators; the
reading instructions in `render()` are outside that budget. Memory notes are
packed first. An exchange too large for the remaining budget is skipped whole.

Exchange mode widens enabled vector/text turn pools to at least 24 and fact-source
turns to at least 16, then groups adjacent messages using at most three directed
`FOLLOWED_BY` edges inside their owning session or agent. It preserves the
original source text in `turns`; displayed excerpts still follow `turn_chars`.
Disabled lanes and zero widths stay disabled. Group membership is available in
`Recalled.exchanges` and serialized when present.

Optional supplemental search looks for problem-resolution episodes in at most
four vector-selected owned sessions when the question asks about problems or
fixes. It shares the text lane's vote and reserves at most a quarter of the
remaining row budget for new exchanges. Up to 10,000 global Action documents it
reuses the main text lane's corpus count for indexed search. Above that, or if
the count is unavailable, local BM25 examines at most 512 documents with 8,192
characters per ranking prefix and a 2 MiB character budget. These prefixes are
ranking features; displayed sources are fetched in full. Exceeding the local
work cap keeps the primary candidate pool. Missing vector/text lanes also skip
supplemental search. Query errors propagate to the caller. Retrieval uses exact
owned entity vectors and requires no extraction or graph reconstruction.

Exchange packing remains opt-in: a [400-question BEAM prototype run](../eval/recall-exchanges.md)
scored 65.46% versus the historical 63.60%, with 37.9% fewer context tokens, but
abstention and multi-session reasoning regressed. The official benchmark numbers
still describe the original configuration. The global count is per request rather
than a transactional snapshot; concurrent inserts may change the indexed
supplemental candidate pool. In multi-space recall the budget applies per space;
it is not a global ceiling over the fused result.

Harnesses reach it as the `recall` tool: sessions-graph registers it under
memgraph-context-graph's `memgraph_context_graph.tools` entry point
(`sessions_graph.tool:RECALL`), and `memgraph-context-graph mcp` serves it, taking
the user and the `[recall]` overrides from the config file. See
[memgraph-context-graph § Recall](../memgraph-context-graph/README.md#recall-memory-for-the-harnesss-model).

With [memory files](#memory-files), recall also searches them: the
`memories` lane matches their passages by vector and their text by full-text
search. Their passages are embedded when the file is reconciled. Matching files
come first as `MEMORY [path, updated date]` rows, ahead of the turns, with one
extra reading rule. A user with no memory files gets exactly the five
conversation lanes' output, header included. That's how the benchmark, whose
corpora have none, stays byte-identical.

## Embeddings for recall

Recall searches a user's history by vector as well as by text, so three
units get vectors, computed **inside Memgraph** by MAGE's `embeddings`
module (no model runs on the host):

| Unit | Text embedded |
|---|---|
| user and assistant messages (`:Action`) | `text`, split into passages of at most 1,200 characters, one vector each in `passage_embeddings` |
| entities | their `text`, for every entity mentioned in one of the session's chunks |
| extracted edges | `"<head> <type> <tail>. <r.text>"`: the fact and the sentence it was read from |

Each vector also records its `embedding_model`. A vector from a different
model counts as missing and is replaced, so vectors from two models never
mix. The default model is `BAAI/bge-small-en-v1.5` (384 dimensions); change
it with:

```bash
memgraph-context-graph config set recall.embedding_model <huggingface-model-name>
```

Recall prepends bge's retrieval instruction to the question only for bge's
English models; another model gets no prefix unless `[recall] query_instruction`
sets the one it was trained with.

When it runs:

- **Turn end and session end:** `SessionsGraphConnector` always spawns a
  detached `sessions-graph embed --session <id>`, whether or not
  `auto_reconcile` is on. Embedding needs no LLM, and the hook never waits on
  the model. Embedding at every turn end keeps recall current mid-session,
  and covers runtimes that never report a session end.
- **Reconciliation:** after extraction, the session's new entities and edges
  are embedded. An embedding failure never fails the reconciliation.
- **Catch-up:** `sessions-graph embed --pending` embeds every session whose
  embedding failed, never ran, or used another model or passage split.

The outcome is on the Session: `embedding_status` is `completed` (with
`embedding_model` and `embedding_scheme`) or `failed` (with `embedding_error`).

This needs Memgraph with MAGE (`memgraph/memgraph-mage`), with 2 GiB of
memory or more for the default model. The model downloads inside Memgraph on
first use. On plain `memgraph/memgraph` every session records `failed`, and
`memgraph-context-graph doctor --connector sessions-graph` reports the
`embeddings` check as failing.

```bash
sessions-graph embed --session s-abc123
sessions-graph embed --pending --limit 50 --model BAAI/bge-small-en-v1.5
```

## API reference

| Method | Description |
|---|---|
| `setup()` | Create constraints, text index, and reconciliation indexes. Run once on first use. |
| `save_memory(user_id, content, *, session_id, memory_id, path)` | Persist a new Memory at `path` (default `/memories/notes/<memory_id>.md`). Returns the stored `Memory` object. |
| `memory_store(user_id, *, session_id=None)` | The memory tool's file commands over the user's Memories. See [Memory files](#memory-files). |
| `get_memories(user_id)` | Return all Memories for a user, newest first. |
| `get_memories_for_session(session_id)` | Return all Memories produced by a session, newest first. |
| `search_memories(user_id, query, *, limit=10)` | Full-text search over Memory content. |
| `update_memory(memory_id, content)` | Replace the content of an existing Memory. Returns `None` if not found. |
| `delete_memory(memory_id)` | Remove a Memory and all its relationships. |
| `recall(user_id, question, *, config=None, model=DEFAULT_EMBEDDING_MODEL)` | What the user's own sessions hold about `question`, as `Recalled` (turns and facts; `lines()`, `render(today)`, `to_json()`). See [Recall](#recall). |
| `record_session_start(session_id, *, user_id, project_key)` | MERGE the Session, owned by `user_id` and in `project_key` when known. The connector calls it on session start. |
| `reconciler` | The graph's `Reconciler`, below: extraction and embedding over the same connection. |

`graph.reconciler` (`sessions_graph.reconciler.Reconciler`) turns sessions and
memory files into entities and vectors, and is the one writer of their
processing status (`reconciliation_*`, `reextract*`, `embedding_*`,
`extraction_status`):

| Method | Description |
|---|---|
| `async reconcile(session_id, *, extraction_backend=None, promote_labels=False, enforce_ontology=False, ontology_path=None, embedding_model=DEFAULT_EMBEDDING_MODEL, reextract=False)` | Reconcile one session, then embed what it wrote (see [Embeddings for recall](#embeddings-for-recall)). Extracts with GLiNER2 under the user's adopted ontology version unless `extraction_backend` overrides it. No LLM. Returns a `ReconciliationSummary`. Requires the `reconciliation` extra. |
| `async reconcile_batch(session_ids, *, lightrag_wrapper, ...)` | Reconcile many sessions with LightRAG as one processing pass. |
| `async reconcile_memory(memory_id, *, extraction_backend=None, enforce_ontology=True, embedding_model=DEFAULT_EMBEDDING_MODEL)` | Re-extract one memory file, replacing what its previous text produced. See [Session reconciliation](#session-reconciliation). |
| `mark_pending(session_id)` | Queue the session for the next `reconcile --pending` sweep. |
| `pending_sessions(*, limit=100)` | Session IDs marked `reconciliation_status = 'pending'`. |
| `pending_memories(*, limit=100)` | Memory IDs whose file changed since it was last reconciled. |
| `embed(session_id, *, model=DEFAULT_EMBEDDING_MODEL)` | Embed the session's messages, entities and edges that lack a vector from `model`, inside Memgraph. Returns counts as `Embedded`; records the outcome on the Session. Raises `EmbeddingUnavailableError` without MAGE or when the model can't load. |
| `pending_embeddings(*, model=DEFAULT_EMBEDDING_MODEL, limit=100)` | Session IDs whose embedding failed, never ran, or used another model. |
| `session_document(session_id, user_id)` / `session_user(session_id)` / `backend_for(model)` | What derivation reads a session as, the session's user, and the cached GLiNER2 backend for an ontology model. |
