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
| Source | LongMemEval v1 + V2, converted | written for this project |
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

## Reconciliation

Injection stages raw turns; **reconciliation** is what turns them into memory —
the same pass a real harness session gets: one LLM call extracting entities into
`Chunk`s (semantic) and a second producing the session's `Episode` (episodic).
Retrieval is therefore scored against the genuine emerged graph, not a shortcut
built for eval.

It is a separate step from injection because it is LLM-backed and slow; folding
it in would make staging a batch cost as much as scoring one.

```python
from context_graph_eval.reconcile import reconcile_batch

result = await reconcile_batch(db, limit=50)   # bounded chunks
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
result.retrieval_context   # rows the graph returned -> scored, and token-counted
result.queries             # what it actually asked -> makes a score diagnosable
result.errors              # failed queries, recorded rather than raised
```

Writes are refused outright. Retrieval must not be able to alter the graph it is
scored against — the same reasoning that keeps the corpus in git rather than in
Memgraph. The step budget is bounded for a related reason: retrieval cost is
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

```python
from context_graph_eval.scoring import aggregate, build_metrics, to_test_case

report = aggregate(scored)
report.by_tier[1].coverage_rate
report.by_tier[1].median_efficiency_tokens   # median, not mean: one pathological
                                             # payload shouldn't move the number
                                             # compared across schema versions
report.by_tier[1].abstention_correct         # reported apart -- here a confident
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

Session ids are namespaced per question (`<question_id>--<session_id>`) because
upstream reuses distractor sessions across questions — 3,942 of 23,867 haystack
ids in the real dataset are duplicates, which would otherwise MERGE different
questions' sessions onto one node.

## Building the Tier 1 corpus

```bash
uv run --package context-graph-eval context-graph-eval build-corpus \
    --limit 60 --out context-graph/eval/corpus/tier1-longmemeval.jsonl
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

The `oracle` variant is refused: it ships evidence sessions only, so retrieval
faces no distractors and both precision and payload-size efficiency would score
well by construction.

## Development

```bash
uv sync
uv run --package context-graph-eval --extra test pytest context-graph/eval/tests
```
