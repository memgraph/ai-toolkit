# Context Graph Eval

The evaluation loop for the [Context Graph](../) family — the instrument that
measures whether raw agent activity actually emerges into knowledge that can be
recalled usefully.

Design decisions live on
[Map: Context-graph emergence pipeline](https://github.com/memgraph/ai-toolkit/issues/297);
this README only summarises what the code does.

## Two tiers, scored separately

| | Tier 1 — adopted | Tier 2 — authored |
|---|---|---|
| Source | LongMemEval v1 (`s` / `m` variants), converted | written for this project |
| Asks | does recall work *mechanically*? | does it work for *what we are building*? |
| Role | regression net | what promotion decisions hang on |

The tiers are never blended into a single headline score. A schema change can
lift personal-memory recall while degrading organizational recall, and one
averaged number would report that as flat or improved.

## Corpus

Questions are stored as [deepeval](https://github.com/confident-ai/deepeval)
`Golden` records serialized to JSONL and committed to git — deliberately *not*
in Memgraph. The corpus is the answer key, and the schema-evolution loop's whole
job is mutating the graph; an answer key living in that graph could be silently
invalidated by a migration, leaving no way to tell a real regression from a
corrupted fixture. Git also proves the corpus did not change between two runs.

Upstream benchmark content is **fetched and converted** against a pinned
release. Only converted output is committed, never vendored raw datasets.

Benchmark survey and licence findings:
[`docs/research/2026-08-memory-benchmarks.md`](https://github.com/memgraph/ai-toolkit/blob/research/memory-benchmarks/docs/research/2026-08-memory-benchmarks.md).

## Running a batch

```bash
docker run -d --name ai-toolkit-eval-memgraph -p 7689:7687 \
    memgraph/memgraph-mage:latest --schema-info-enabled=true

uv run --package context-graph-eval context-graph-eval run \
    --limit 100 --judge-model anthropic:claude-sonnet-4-5-20250929
```

`--judge-model`/`--agent-model` take a `provider:model_id` spec (e.g.
`anthropic:claude-sonnet-4-5-20250929`, `openai:gpt-4o`); a bare model id
keeps that role's default provider (#329 -- the judge stays on a different
provider from the pipeline by default, but a shared provider is a legitimate
experiment, flagged loudly rather than refused).

`--extraction-backend {lightrag,gliner2}` (default `lightrag`) picks what
reconciliation uses to extract entities -- `gliner2` is local and LLM-free
(`unstructured2graph.gliner2_backend.GLiNER2Backend`), needs `pip install
'gliner2[local]>=2.0.0'` installed manually (see that module's own
docstring), and still needs an LLM key regardless: narrative summarization
has no GLiNER2 equivalent and always runs through the LightRAG wrapper's LLM.
`RunMeta` records which backend built a saved run, and `compare()` refuses
across a mismatch the same way it refuses across a judge or tokenizer
mismatch.

The runner owns the **pipeline** loop; deepeval owns the **scoring** loop
underneath it:

```
runner  ->  inject -> reconcile -> retrieve  ->  deepeval  ->  metrics
```

deepeval knows nothing about the first three stages, and all of them must
happen before an `actual_output` exists to score. Ordering is the runner's real
job: retrieving before injection would query an empty graph and score every
question a miss, while scoring before reconciliation would score raw turns
rather than emerged memory — the thing actually under test.

`--skip-reconcile` reuses the already-reconciled graph as-is: no wipe, no
re-injection, no distillation. Reconciliation dominates run cost, so iterating
on retrieval or scoring shouldn't pay for it twice.

It **refuses** rather than trusting what it finds — if the graph does not hold
this run's sessions, or holds them still pending distillation, it stops. Both
cases are silent otherwise, and both end in every question scoring as a recall
miss that gets reported as an ordinary result.

The runner **refuses to start** if `CONFIDENT_API_KEY` is set: deepeval uploads
a test run whenever a Confident AI key is present, and eval results stay local
for the same owned-IP reason the corpus does.

## Comparing runs

Promotion is human-gated (#299), so the report's job is not to decide — it is
to make the decision *makeable*.

```bash
context-graph-eval run --limit 100 --save runs/baseline.json --label baseline
# ...change something...
context-graph-eval run --limit 100 --save runs/candidate.json --label candidate \
    --changed "decay rule v3 (7-day window -> usage-based)"

context-graph-eval compare runs/baseline.json runs/candidate.json --noise-floor 4
```

```
VERDICT  improved
  noise floor +/-4pp

Tier 1              base    cand   delta
  coverage            12/20  13/20     +5pp  REAL
  efficiency med     1,840   1,120      -39%
  improvements     q_12
```

Two behaviours matter more than the layout:

**It refuses to compare runs measured differently.** A different corpus
revision, judge model, or tokenizer makes two runs incomparable — #302 and #304
pinned those precisely so a comparison would mean something. Comparing across
pins measures the pin change as though it were the change under test, and
reports it confidently. That is a refusal, not a warning.

**Without calibration it will not call anything real.** Judged scores vary run
to run, so a bare `12/20 -> 13/20` invites reading a win into noise. The noise
floor comes from #304's repeat-and-compare check; absent one, the report says
`NOT CALIBRATED` and returns `inconclusive` rather than guessing.

A real coverage regression decides the verdict even when efficiency improved —
coverage is the gate (#309), and a cheaper answer missing facts is not a better
one. Efficiency alone never declares an improvement, since "coverage held"
cannot be established inside the noise floor.

> Sizing caveat: at 100 questions, one question = 1pp, down from 5pp at the
> old 20-question corpus. Closer to a plausible noise floor, not below it
> (#304's repeat-and-compare found ±5pp on 6 questions) — confirm calibration
> on the current corpus size before trusting a small coverage delta.

## The gold slice

Tier 1 injects fixture text straight into the graph — fast and cheap, but it
never exercises the capture layer. The gold slice runs a **real Claude Code
session** with hooks live, so what gets scored is what the pipeline actually
records. It is the only eval coverage `skills-graph` and subagent nesting get at
all; #308 found no benchmark covers either.

**One question, for now.** Each is a real, billed session, so the slice grows by
*carrier* — where the fact physically lives in the graph — not by round number.

The first carrier is a fact that exists **only inside a subagent**, because:

- top-level recall is already covered by Tier 1's 100 questions, and
- the nested carrier has a demonstrated silent-failure mode. #281 found
  `get_session_actions()` does a single-hop `HAS_ACTION` match, so once subagent
  activity moved under `(:Agent)`, reconciliation would stop seeing it — no
  entities, no Episode mention, **no error**. Caught once by reading code; as a
  gold-slice question it is caught automatically, forever.

Three constraints shape the planting prompt, and each is load-bearing:

| Constraint | Why |
|---|---|
| Prompt must not contain the fact | otherwise it lands top-level too, and recall passes with nesting fully broken |
| Fact must be unguessable | a fact answerable from priors tests nothing; hence a hex revision, not "MIT" |
| No literal `SKILL.md` path | `SkillGraphConnector` records a skill read that never happened (#293) |

`evidence_is_nested()` is checked **before** trusting a recall result: if the
model declines to delegate, the fact lands top-level and recall succeeds
trivially — a false pass indistinguishable from a real one.

### Driving the session

```bash
context-graph-eval gold-slice --memgraph-url bolt://localhost:7689
```

Pointing hooks at the eval instance is the whole difficulty. Config *values*
come only from the config file (ADR 0002), so the driver cannot just export
`MEMGRAPH_URL`. Backing up and rewriting the user's real config would work but
is hostile: that file is a single global, so a crash mid-run leaves their normal
sessions pointed at the eval instance.

Instead the driver writes a throwaway config and names it with
`CONTEXT_GRAPH_CONFIG` (ADR 0003) in the environment of the subprocess it
spawns. Nothing ambient is set and the user's own config is never touched.

The session is **real and billed**, so the run prints its transcript whether or
not it succeeded: a zero exit means the CLI did not crash, not that the model
delegated or that hooks recorded anything, and finding out otherwise costs
another session.

## Reconciliation

Injection stages raw turns; **reconciliation** is what turns them into memory —
the same pass a real harness session gets, producing `Chunk`s and entities
(semantic) plus the session's `Episode` (episodic). Retrieval is therefore
scored against the genuine emerged graph, not a shortcut built for eval.

It is far more expensive than "a call or two per session". Measured on 39
sessions: extraction runs per chunk, in two passes each (an initial extraction
and a gleaning pass), which came to **~46 LLM calls per session** — and ~89
before the chunk sizing was fixed. Reconciliation is ~97% of all LLM calls in a
run, which is why `--skip-reconcile` exists and why repeat runs are minutes
rather than hours.

46/session was measured on only 39 sessions — likely a **floor**, not a flat
rate, at batch scale. LightRAG pays for an LLM merge-summary call once an
entity/relation hits `force_llm_summary_on_merge` (default 8) raw mentions.
Confirmed by reading the merge path: that description list rebuilds from
*all* historical mentions (capped at `max_source_ids_per_relation`/`_entity`,
default 200) every merge, not just the first. Eval batch = one **shared**
workspace across sessions (needed for retrieval distractors), so a recurring
entity — a name, a repeat topic — re-pays this cost on every later session
that mentions it again. `context_graph_eval.reconcile` raises the eval-only
threshold to 30 (`_resolve_reconciliation_tuning`, `setdefault`-only — never
touches production `sessions-graph` or an operator's own value). Trade: plain
concatenation instead of an LLM summary for entities that stay under 30.

### Extraction granularity: session-batching + the embedding ceiling

`sessions-graph` reconciles a whole session as one document, not one per turn
(map #297). Before: each turn extracted isolated from the rest of the
session — hid cross-turn facts (coreference, a fact stated in one turn and
referenced later) and cost one LightRAG document per turn. Real extraction
granularity = whatever re-chunks the combined text smallest, not turn count.

Not LightRAG's own `CHUNK_SIZE` (1200 tokens): LightRAG re-splits any chunk
down to the embedder's `max_token_size` *before* embedding, and extraction
runs on the re-split pieces. `lightrag-memgraph`'s default embedder
(Memgraph's local `all-MiniLM-L6-v2`, picked for zero external cost) has
`max_token_size=256` — near this corpus's ~245-token average turn, leaving
little room to consolidate. `context_graph_eval.reconcile._eval_embedding_func`
swaps in `BAAI/bge-m3` (still local, same Memgraph `embeddings` module —
confirmed against a live instance: dim 1024, max seq length 8192) for eval
batches, raising the ceiling above all but the largest sessions in the corpus.

Measured on the same 5 real sessions at each step:

| | extraction+gleaning calls | calls/session |
|---|---|---|
| Per-turn (original) | 106 | 21.2 |
| Session-batched, 256-token ceiling | 70 | 14.0 |
| Session-batched, bge-m3 (8192-token ceiling) | **18** | **3.6** |

**5.9x** reduction end to end, zero quality trade-off (bge-m3 is a strict
upgrade over all-MiniLM, not a cheaper substitute) — unlike the
merge-threshold change above, which trades quality for cost.

Caveat swapping embedders mid-project: lightrag-memgraph's vector storage
creates its Memgraph vector index once, and treats a second `CREATE VECTOR
INDEX` as "already exists" no matter *why* creation failed — dimension
mismatch included. `inject.py`'s `_wipe()` now drops every existing vector
index before a batch, not just the graph's nodes, so a batch that changes
embedding model is as safe as one that doesn't.

It is a separate step from injection because it is LLM-backed and slow; folding
it in would make staging a batch cost as much as scoring one.

```python
from context_graph_eval.reconcile import reconcile_batch

result = await reconcile_batch(db, limit=50)  # bounded chunks
```

Partial failure is reported rather than raised: a score only means something if
you know how much of the graph is actually populated, so one session that can't
be distilled must not abandon the rest of the batch.

LLM credentials resolve from context-graph's config file (ADR 0002) before
falling back to the environment, so eval runs standalone without exported
variables.

## Retrieval

The v1 baseline is deliberately the *existing* query surface: an agent gets the
graph schema and writes its own read-only Cypher. No ranking, no query
templates, no vector search — those are deferred until this baseline's failures
say what they should be, since building them first means designing against a
score nobody has seen.

```python
from context_graph_eval.retrieval import DeepEvalLLM, ReadOnlyGraph, retrieve

result = await retrieve(question, graph=ReadOnlyGraph(db), llm=DeepEvalLLM(model))
result.retrieval_context  # rows the graph returned -> scored, and token-counted
result.queries  # what it actually asked -> makes a score diagnosable
result.errors  # failed queries, recorded rather than raised
```

Writes are refused outright. Retrieval must not be able to alter the graph it is
scored against — the same reasoning that keeps the corpus in git rather than in
Memgraph.

**LightRAG's own storage labels are refused too.** It persists its KV, vector,
doc-status and *LLM response cache* into the same graph, and that cache contains
the answers — an agent querying it scores coverage having exercised none of the
graph model. Refused rather than merely hidden from the schema, since hiding a
label does not stop an agent guessing it. The step budget is bounded for a related reason: retrieval cost is
itself scored, so an agent allowed to query indefinitely could buy coverage with
an unbounded payload.

Note the write guard duplicates `mcp_memgraph.servers.server.is_write_query`
rather than importing it — that module binds a client registry to global env
config at import time, which would point retrieval at whatever Memgraph the
environment names instead of the eval instance. If that guard gains a pattern,
this one needs it too.

## Scoring

Quality is judged, cost is counted — asking an LLM to grade a number you can
count adds variance for no information.

- **Coverage** — `ContextualRecallMetric` over retrieval, plus one `GEval`
  rubric over the answer itself.
- **Efficiency** — a deterministic token count of the retrieval payload. Fewer
  tokens returned for the same answer is better.

Coverage is a **hard gate**; efficiency only ranks questions that cleared it.
Otherwise the metric is trivially gamed by returning nothing.

Three refinements, each added because it caught a wrong number:

- **A question answered from an empty retrieval cannot pass.** `ContextualRecall`
  over an empty context is vacuously satisfied, so "not in memory" scored 1.0 on
  a question whose answer was "7 days". Abstention questions are exempt — for
  those an empty payload is correct.
- **Abstention is judged on refusing**, not on reciting the near-miss fact
  upstream pairs with it ("you mentioned your cat Luna but not your hamster").
  Scored on the shared coverage rubric it was unpassable: the agent declined
  correctly 5/8 and scored 0/8.
- **A question the judge could not score is reported `unscored`, never as 0%.**
  A judge outage once printed a confident `coverage 0/2 (0%)`.

```python
from context_graph_eval.scoring import aggregate, build_metrics, to_test_case

report = aggregate(scored)
report.by_tier[1].coverage_rate
report.by_tier[1].median_efficiency_tokens  # median, not mean: one pathological
# payload shouldn't move the number
# compared across schema versions
report.by_tier[1].abstention_correct  # reported apart -- here a confident
# answer is the failure
```

`RunReport` has **no** blended headline field, by design. A single number across
tiers is exactly what would let an organizational-recall regression hide behind
a personal-memory gain.

The efficiency tokenizer is pinned for the same reason the judge model is: a
tokenizer change silently shifts every efficiency number, and two runs measured
differently aren't comparable.

## Isolation

Each eval **batch** runs against a **dedicated Memgraph instance**, cleared
before fixtures load. This is a reproducibility requirement before it is a
hygiene one: comparing two schema versions is meaningless if the graph also
holds whatever ambient sessions happened to land that week.

> Per-batch *databases* were the original plan, but that is Memgraph
> multi-tenancy and requires an Enterprise licence. A dedicated instance gives
> the same known-fixed-state guarantee on a community licence, and clearing it
> is safe precisely because nothing else lives there. **Never point this at a
> shared or development database.**

```bash
docker run -d --name ai-toolkit-eval-memgraph -p 7689:7687 \
    memgraph/memgraph-mage:latest
```

Tests read `EVAL_MEMGRAPH_URL` (default `bolt://localhost:7689`) and skip if no
instance is reachable.

Markers inside a batch are provenance only — never the mechanism keeping
eval-agent traces out of the graph under test. `SessionFixture.holds_evidence`
is corpus-side bookkeeping and is deliberately never written to the graph:
storing it would tell retrieval where the answer lives.

Session ids are kept **verbatim**. Upstream reuses distractor sessions across
questions — 3,942 of 23,867 haystack ids in the real dataset repeat — but *zero*
of those repeats carry differing content, so a repeated id genuinely is the same
session. Letting it become one node matches how a real organizational graph
would hold it; namespacing per question would store byte-identical copies and
pay to reconcile each. The duplicate-turns hazard that implies is handled by
deduplicating at injection instead.

## Building the Tier 1 corpus

```bash
uv run --package context-graph-eval context-graph-eval build-corpus \
    --limit 100 --out context-graph/eval/corpus/tier1-longmemeval.jsonl
```

Fetches a **pinned** LongMemEval revision, converts it, and writes the JSONL
that gets committed. The downloaded upstream file is a build artifact and is
never committed. Bumping `--revision` invalidates prior baselines, the same way
bumping the judge model does.

`--limit` samples deterministically, stratified by `(question_type, abstention)`
and proportional to upstream with a small floor per stratum, so that:

- a regenerated corpus produces no spurious diff, and two runs stay comparable;
- the aggregate score reflects the real distribution rather than over-weighting
  rare categories;
- no category rounds to zero and vanishes silently.

> **Floor dominates at small sizes.** 10 populated strata, floor=2, so
> anything at or below `--limit 20` gets exactly 2 per stratum regardless of
> real size — proportionality gone. That's what the old 20-question corpus
> did: 40% abstention vs upstream's 6%, a 6.7x over-weight that moved the
> headline ~16pp. Corpus now built at `--limit 100` — floor only mildly nudges
> the 3 smallest strata (true share ~1.2%), abstention lands at 8%.
>
> **A prefix of the corpus is also sampled, not just the whole file.** `run
> --limit N` (below) reads the committed corpus and takes `corpus[:N]` (#302)
> instead of re-deriving a fresh sample — must, so two compared runs provably
> ask the same questions. But `build_corpus`'s old round-robin order put one
> record per stratum each pass, so *any* prefix reproduced the same
> floor-uniform distortion one layer later — `run --limit 20` against a
> proportional 100-question corpus would silently re-run the old skewed 20.
> `build_corpus` now shuffles with a fixed seed before returning (still
> byte-identical across regens), so a prefix is an unbiased sample, not a
> systematic bias. A prefix well below the full corpus size still carries more
> sampling noise than the aggregate — treat `run --limit` below the committed
> size as a cheap smoke check, not a representative score.

## Known limitations

The loop reliably finds problems; it cannot yet **rank** two versions.

- **The passing set is unstable.** Three repeats against an identical graph, at
  `temperature=0` throughout, gave `0 of 6` questions passing in all three, with
  two runs' passing sets entirely disjoint. The coverage rate looked steady
  (15%, 10%, 15%) while the questions beneath it churned completely. Root
  cause, confirmed directly (#324): not the judge -- a fixed test case scored
  by the same judge repeatedly is perfectly stable -- but the retrieval
  agent's own Cypher-query generation, which is not deterministic even at
  `temperature=0` (a documented limitation of hosted LLM APIs). That changes
  `retrieval_context`'s content and literal rendering run to run against the
  exact same frozen graph, which flips `ContextualRecallMetric` even when the
  final answer text never changes.
- **The noise floor is most of the signal** — ±5pp against a ~13% mean.
- **The memory tier is ~14:1 assistant-sourced.** Assistant turns contain far
  more nameable things than user turns, so entity extraction is dominated by
  what the model said rather than what the user did. The eval questions all ask
  about user facts.
- **Tier 2 has one question.**

`calibrate` reports set stability alongside the floor, and now each
question's own pass rate across the repeats -- naming the specific flaky
ones, since the aggregate ratio alone cannot tell a question that always
fails from one that passes half the time. `compare` refuses to report an
efficiency delta when the two runs share no passing question. All three
exist because the aggregate alone hid what was actually happening.

The `oracle` variant is refused: it ships evidence sessions only, so retrieval
faces no distractors and both precision and payload-size efficiency would score
well by construction.

## Development

```bash
uv sync
uv run --package context-graph-eval --extra test pytest context-graph/eval/tests
```
