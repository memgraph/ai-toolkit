# Research: what does sessions-graph's existing reconciliation actually write today?

- Ticket: [memgraph/ai-toolkit#260](https://github.com/memgraph/ai-toolkit/issues/260)
- Parent map: [memgraph/ai-toolkit#259](https://github.com/memgraph/ai-toolkit/issues/259)
- Method: primary-source read of the code paths below, end to end. All claims cite `path:line`. No secondary paraphrase — the sessions-graph README/CONTEXT.md are cited only where the code was independently verified to match them.

Primary sources read in full:
- `context-graph/sessions-graph/src/sessions_graph/core.py`
- `context-graph/sessions-graph/src/sessions_graph/cli.py`
- `context-graph/sessions-graph/src/sessions_graph/reconciliation.py`
- `context-graph/sessions-graph/src/sessions_graph/models.py`
- `context-graph/sessions-graph/src/sessions_graph/connector.py`
- `context-graph/sessions-graph/CONTEXT.md`
- `context-graph/sessions-graph/README.md`
- `unstructured2graph/src/unstructured2graph/loaders.py`
- `unstructured2graph/src/unstructured2graph/memgraph.py`
- `unstructured2graph/src/unstructured2graph/ontology.py`
- `unstructured2graph/src/unstructured2graph/__init__.py`
- `context-graph/actions-graph/src/actions_graph/models.py`, `core.py` (`get_session_actions`)
- `context-graph/skills-graph/src/skills_graph/core.py`, `connector.py`
- `context-graph/sessions-graph/tests/test_e2e_reconciliation.py` (used only to corroborate what labels/edges actually land, not as a source of new claims)
- `.github/workflows/*.yaml`, `scripts/dev-memgraph.sh` (grepped for automation)

All line numbers refer to the repo state at commit `90c7719` (`origin/main`, "ci(unstructured2graph): download NLTK data explicitly (#258)").

---

## 1. Does `reconcile_session` ever write a `(:Memory)` node? What does it actually write, and how does it connect back?

**No. `reconcile_session` never creates, updates, or touches a `(:Memory)` node.** The only sessions-graph method that creates `(:Memory)` nodes is `save_memory`, an entirely separate, synchronous, explicit write path (`context-graph/sessions-graph/src/sessions_graph/core.py:90-148`), called directly by application/agent code, not by reconciliation.

`reconcile_session` (`core.py:269-389`) does the following, and nothing else:

1. Reads session content — never writes Action/Memory/Session content itself, only reads it:
   - `actions_graph.get_session_actions(session_id)` (`core.py:340`) — all `(:Action)` nodes under `(:Session)-[:HAS_ACTION]->(:Action)` for the session (`context-graph/actions-graph/src/actions_graph/core.py:594-624`).
   - `self.get_memories_for_session(session_id)` (`core.py:341`, method at `core.py:172-191`) — existing `(:Memory)` nodes already reachable via `(:Session)-[:PRODUCED_MEMORY]->(:Memory)`. This is a *read* of Memories written earlier via `save_memory`; reconciliation never creates a Memory itself.
2. Builds `ReconciliationSource` items from that content via `build_reconciliation_sources` (`reconciliation.py:120-138`), dedupes by SHA-256 content hash (`core.py:344-346`, `content_hash` at `reconciliation.py:34-36`).
3. Hands the deduped text list to `unstructured2graph.from_texts(...)` (`core.py:349-358`, imported at `core.py:335`).
4. Links each resulting Chunk back to its source node via `HAS_CHUNK` (`core.py:360`, `_link_chunks_to_sources` at `core.py:404-434`).
5. Sets `s.reconciliation_status = 'completed'|'failed'` and `s.reconciled_at`/`s.reconciliation_error` on the `(:Session)` node (`core.py:362-368`, `376-382`) — the only node reconciliation mutates directly other than Chunk/entity nodes created by unstructured2graph.

What `unstructured2graph.from_texts` actually writes (traced into `unstructured2graph/src/unstructured2graph/loaders.py` and `memgraph.py`):

- **`(:Chunk {hash, text})`** — one node per chunk of each input text, produced by `parse_text()` (`loaders.py:76-107`) and upserted via `create_nodes_from_list(..., "Chunk", ..., merge_key="hash")` inside `_ingest_chunks` (`loaders.py:222-226`, `memgraph.py:35-84`, MERGE-keyed on `hash` so re-running is idempotent).
- **Entity nodes under a workspace label**, e.g. `:base` — written by LightRAG itself via `lightrag_wrapper.ainsert(input=chunk.text, file_paths=[chunk.hash])` (`loaders.py:234-236`), *not* by unstructured2graph's own Cypher. unstructured2graph only wires them up afterward:
  - `connect_chunks_to_entities(memgraph, "Chunk", entity_workspace)` (`loaders.py:237`) creates `(entity:{workspace})-[:MENTIONED_IN]->(chunk:Chunk)` by matching `entity.file_path == chunk.hash` (`memgraph.py:87-94`).
  - If `enforce_ontology=True` (the CLI's actual setting — see §2): `promote_entity_types_to_labels` (`memgraph.py:97-131`) *additively* sets a real Memgraph label (e.g. `:Person`) on top of the workspace label for every entity whose `entity_type` matches the ontology (`unstructured2graph/src/unstructured2graph/ontology.py`, `DEFAULT_ONTOLOGY` loaded from `default_ontology.yaml`), and stamps `ontology_conformant = false` on everything else. The workspace label itself is never removed (`memgraph.py:104-105`), since LightRAG's own re-ingestion MERGE depends on it.
  - If `promote_labels=True` instead (and `enforce_ontology=False`): `promote_all_entity_types_to_labels` (`memgraph.py:134-163`) promotes every distinct `entity_type` with no vocabulary gating.
  - Default for both (`sessions-graph reconcile`'s CLI actually passes `enforce_ontology=True`, see §2, so this default path is not what runs in practice, but it's `from_texts`'s own default): no label promotion at all — entities sit under only the workspace label with an `entity_type` property (`loaders.py:273-284`, `303-306`).

**Node labels created/touched by the reconciliation path, exhaustively:** `(:Chunk)` (created), entity nodes under `:{workspace}` plus ontology labels like `:Person`/`:Organization` (created by LightRAG + promoted by unstructured2graph), `(:Session)` (only its `reconciliation_status`/`reconciled_at`/`reconciliation_error` properties set — the node itself is expected to pre-exist from `SessionsGraphConnector`). **`(:Memory)` is never among them.**

**Connection back to Memory/Session/Action/User:** Chunks connect to their originating `(:Action)` or `(:Memory)` node via `(:Action|:Memory)-[:HAS_CHUNK]->(:Chunk)` (`core.py:404-434`, `NODE_LABELS` map at `reconciliation.py:103-106` mapping `kind="action"→(Action, action_id)` / `kind="memory"→(Memory, memory_id)`). This is the *only* edge reconciliation adds that touches a `(:Memory)` node — it links an already-existing Memory to a derived Chunk, it does not modify the Memory or create a new one. From there, provenance to `Session`/`User` is inherited transitively through pre-existing edges that reconciliation itself never writes: `(:Session)-[:HAS_ACTION]->(:Action)` (owned by actions-graph) and `(:Session)-[:PRODUCED_MEMORY]->(:Memory)` / `(:User)-[:HAS_MEMORY]->(:Memory)` / `(:User)-[:HAD_SESSION]->(:Session)` (owned by sessions-graph's `save_memory` and `SessionsGraphConnector._on_session_start`, `connector.py:126-147`). Entity nodes connect to Chunks via `MENTIONED_IN` (`memgraph.py:87-94`), and only reach a Session/User/Memory by walking `entity -[:MENTIONED_IN]-> chunk <-[:HAS_CHUNK]- (Action|Memory)`.

This matches (and is independently verified against) `context-graph/sessions-graph/CONTEXT.md:40-41,75`, which states explicitly: *"Session Reconciliation... [is] Unrelated to Memory — it is derived and mechanical, not an explicit assertion"* and *"it produces Chunk/entity nodes, never Memory nodes. Do not describe it as 'memory reconciliation' or 'memory extraction'."* The e2e test `context-graph/sessions-graph/tests/test_e2e_reconciliation.py:97-108` corroborates concretely: it asserts `(:Session)-[:HAS_ACTION]->(:Action)-[:HAS_CHUNK]->(:Chunk)` exists (line 99) and `(:{workspace})-[:MENTIONED_IN]->(:Chunk)` exists (line 107), and never asserts anything about `(:Memory)`.

---

## 2. What triggers reconciliation end-to-end today?

**Setting `reconciliation_status='pending'`:** `SessionsGraphConnector._on_session_end` (`context-graph/sessions-graph/src/sessions_graph/connector.py:149-154`) calls `_mark_pending_reconciliation` (`connector.py:156-160`), which runs `SET s.reconciliation_status = 'pending'` synchronously, in-hook, on every `SessionEndEvent`. This is cheap and has no LLM calls, so it's safe to run inside a hook-runtime timeout (`connector.py:74-75`, doc comment).

**What `get_pending_reconciliation_sessions` looks for:** `MATCH (s:Session {reconciliation_status: 'pending'}) RETURN s.session_id ... LIMIT $limit` (`core.py:391-402`) — a simple property-equality scan over `(:Session)`, ordered by `session_id`, capped by `limit` (CLI default 100, `cli.py:65-69`).

**Is `reconcile --pending` ever invoked automatically?**
- Grep of `.github/workflows/*.yaml` (`tests.yaml`, `lint.yaml`, and every `release-*.yaml`) for `reconcile` found **no matches** — no CI workflow runs it.
- No cron config exists anywhere in the repo.
- The only two ways reconciliation runs today, both requiring deliberate action:
  1. **Manual CLI invocation**: `sessions-graph reconcile --session ID` or `sessions-graph reconcile --pending [--limit N]` (`context-graph/sessions-graph/src/sessions_graph/cli.py:34-111`), or via the dev convenience wrapper `scripts/dev-memgraph.sh reconcile` (`scripts/dev-memgraph.sh:64-67,81`), which just shells out to the same CLI.
  2. **Opt-in "auto_reconcile"**: `SessionsGraphConnector(graph, auto_reconcile=True)`, or the env var `SESSIONS_GRAPH_AUTO_RECONCILE` (truthy values `1`/`true`/`yes`/`on`, `connector.py:58-62`) read at connector construction when the kwarg isn't passed. If enabled, `_on_session_end` additionally calls `_spawn_reconciliation` (`connector.py:153-154,162-176`), which `subprocess.Popen`s a **detached** (`start_new_session=True`) `sessions-graph reconcile --session <id>` process and does not wait on it. This is fire-and-forget: if the detached process dies (machine sleep, crash) the session just stays `pending` forever until a manual/backfill `--pending` sweep picks it up (`connector.py` docstring at `connector.py:76-79`; README corroborates at `context-graph/sessions-graph/README.md:157-163`). `auto_reconcile` defaults to `False`/off (`connector.py:93`) — it is opt-in, and even when on, it is event-driven-but-detached, not a scheduled/cron sweep. `scripts/dev-memgraph.sh dogfood-env` (`scripts/dev-memgraph.sh:71-73,86-88`) is a dev-only helper that prints the export statements to turn this on for a locally-launched Claude Code session — still manual, still per-developer opt-in, not CI/production automation.

**Conclusion: today, reconciliation is either a strictly manual CLI invocation, or an opt-in, best-effort, detached per-session background process — there is no scheduled/cron/CI-triggered sweep anywhere in this repo.**

---

## 3. What does `ReconciliationSummary` capture, and where does the extracted text actually come from?

**`ReconciliationSummary`** (`context-graph/sessions-graph/src/sessions_graph/reconciliation.py:109-117`) is a frozen dataclass with exactly these fields:

```python
session_id: str
status: str            # "completed" | "failed"
texts_considered: int   # len(sources) before dedup
texts_deduped: int      # len(unique_texts) after content-hash dedup
error: str | None = None
```

It reports **counts only** — no list of which entities/Chunks were produced, no token/cost accounting, no per-source detail. `reconcile_session` constructs it at `core.py:369-374` (success) and `core.py:383-389` (failure, with `error=str(e)`).

**Where the raw text comes from** — traced via `build_reconciliation_sources` (`reconciliation.py:120-138`), which is the actual function name in this codebase for what the ticket calls "extract_enrichable_text":

- **Two sources, both pulled in every reconciliation run: Action content AND Memory content.**
- Actions: for every `Action` returned by `actions_graph.get_session_actions(session_id)` (all action types, no filter — `core.py:340`), `extract_reconcilable_text(action)` (`reconciliation.py:52-85`) is applied:
  - Only `Message`, `ToolCall`, and `ToolResult` instances yield text; every other `ActionType` (`ErrorEvent`, `SubagentEvent`, `StructuredOutput`, `PermissionRequest`, `RateLimitEvent` — enumerated in `context-graph/actions-graph/src/actions_graph/models.py`) returns `None` and is skipped (`reconciliation.py:62-70`).
  - For `Message`/`ToolResult`, text comes from `action.content` via `_content_to_text` (`reconciliation.py:39-49`), which handles both a plain string and Anthropic-style content-block lists (joining `block["text"]` fields).
  - For `ToolCall`, text is `json.dumps(action.tool_input, default=str)` (`reconciliation.py:65-68`) — the tool call's *input arguments*, not its result.
  - Empty/whitespace-only text is dropped (`reconciliation.py:72-73`); anything over `MAX_RECONCILABLE_CHARS` (8000, `reconciliation.py:31`) is truncated with a logged warning (`reconciliation.py:76-83`).
- Memories: every `Memory` from `self.get_memories_for_session(session_id)` (`core.py:341`, all Memories linked to the session via `PRODUCED_MEMORY`) contributes its `content.strip()` verbatim, no truncation applied at this stage since it's checked only for non-empty (`reconciliation.py:135-137`) — though the same `MAX_RECONCILABLE_CHARS` truncation only actually runs inside `extract_reconcilable_text`, which is Action-only, so a Memory longer than 8000 chars is **not** truncated before being sent to `from_texts`.
- Both source kinds land in one ordered `list[ReconciliationSource]` (`reconciliation.py:130-138`), each tagged `kind="action"` or `kind="memory"` plus the originating node's ID, so results can be traced back (`NODE_LABELS` at `reconciliation.py:103-106`).
- `core.py:344-346` then dedupes this combined list by SHA-256 content hash before any of it reaches `unstructured2graph.from_texts` / LightRAG (`core.py:348-358`) — so the LLM extraction pipeline's actual input is: deduplicated, truncated (Actions only) text drawn from **both** Action content (Messages/ToolCalls/ToolResults) and Memory content, for one session at a time.

---

## 4. Is there any existing mechanism for detecting repeated patterns across sessions/time (a precondition for skill mining)?

**No such mechanism exists anywhere in this codebase.** This was checked by grepping `context-graph/sessions-graph/src`, `context-graph/actions-graph/src`, `context-graph/skills-graph/src`, and `unstructured2graph/src` for `pattern|frequency|cluster|similar|duplicate|repeat` (case-insensitive). Every hit found is unrelated to cross-session/cross-time pattern detection:

- All `pattern` hits are either regex-pattern validation error messages (e.g. `context-graph/sessions-graph/src/sessions_graph/models.py:32,38`, `context-graph/actions-graph/src/actions_graph/models.py:47,54`) or unrelated uses of the English word: `SkillGraph.search_by_name(self, pattern: str)` (`context-graph/skills-graph/src/skills_graph/core.py:256-273`) is a case-insensitive **substring** match (`WHERE toLower(s.name) CONTAINS toLower($pattern)`, `core.py:261`) against already-registered Skill names — not pattern *mining*. `skills_graph/connector.py:209` uses `"pattern"` only as one of several dict keys (`"name"`, `"skill_name"`, `"skill"`, `"pattern"`) it checks when trying to pull a skill name out of a tool call's input — again unrelated to repetition detection.
- All `duplicate` hits are about idempotent MERGE-vs-CREATE node upserts (`unstructured2graph/src/unstructured2graph/memgraph.py:49,196` — "re-running... duplicates them" / constraint "rejects duplicate values") — schema-level duplicate *prevention*, not behavioral duplicate/pattern *detection*.
- No hits at all for `frequency`, `cluster`, `similar`, or `repeat`.

The closest existing analog is **exact-match content-hash dedup** inside a single `reconcile_session` call (`content_hash` / `unique_texts` at `reconciliation.py:34-36`, `core.py:344-346`) — but this only collapses byte-identical text *within one session's reconciliation batch* before billing the LLM; it has no concept of near-duplicate/similar text, no cross-session or cross-time window, and produces no aggregate/frequency signal that could seed skill mining.

`skills-graph`'s `record_skill_usage` (`context-graph/skills-graph/src/skills_graph/core.py:158`) does record when an *already-registered* Skill is invoked (so usage counts of *known* skills could in principle be queried after the fact), but this presupposes a Skill already exists — it is not a mechanism for discovering a new reusable skill from repeated raw behavior, which is what "mining" in the parent map's framing requires.

**Plainly: no repeated-pattern/frequency/clustering detection mechanism for skill mining exists today, in sessions-graph, actions-graph, skills-graph, or unstructured2graph.** This is a genuine gap, not a tangential feature that stretches to fit.

---

## Summary for the parent map (#259)

1. Reconciliation writes `(:Chunk)` nodes and LightRAG entity nodes (under a workspace label plus optional ontology-gated labels like `:Person`); it **never** writes `(:Memory)`. It links Chunks back to the `(:Action)`/`(:Memory)` node they came from via `HAS_CHUNK`, and only reaches `Session`/`User` transitively through pre-existing edges it doesn't itself write.
2. `SESSION_END` → `SessionsGraphConnector` synchronously sets `reconciliation_status='pending'` (no LLM call). Actual extraction is either a strictly manual CLI call (`sessions-graph reconcile --session|--pending`) or an opt-in, best-effort, detached-subprocess `auto_reconcile`. No CI/cron ever invokes it.
3. `ReconciliationSummary` is counts-only (`session_id`, `status`, `texts_considered`, `texts_deduped`, `error`). Input text comes from **both** Action content (Message/ToolCall/ToolResult, truncated at 8000 chars) and Memory content (untruncated), deduped by content hash before hitting the LLM.
4. No repeated-pattern/frequency/clustering mechanism exists anywhere in this codebase — a real gap for any future skill-mining design, not something to force-fit from an existing feature.
