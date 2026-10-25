# Reconciliation, ontologies, and embeddings

[Back to Sessions Graph](../README.md)

## Session reconciliation

Session reconciliation runs a session's Actions Graph content (Messages,
ToolCalls, ToolResults) through [unstructured2graph](../../../unstructured2graph)
as one document, extracting entities with local GLiNER2 under the user's
[ontology version](#ontology-versions) (or another `ExtractionBackend` passed
as `graph.reconciler.reconcile(..., extraction_backend=...)`), and links them back to
the session that produced them. See [`CONTEXT.md`](../CONTEXT.md#language) for
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
  `ontology_conformant = false`. See [entity typing](../../../unstructured2graph/README.md#entity-typing--ontology).

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
[unstructured2graph § entity typing](../../../unstructured2graph/README.md#entity-typing--ontology).

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
[hygm](../../../hygm/README.md)):

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
