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

## Scoring

Quality is judged, cost is counted — asking an LLM to grade a number you can
count adds variance for no information.

- **Coverage** — `ContextualRecallMetric` over retrieval, plus one `GEval`
  rubric over the answer itself.
- **Efficiency** — a deterministic token count of the retrieval payload. Fewer
  tokens returned for the same answer is better.

Coverage is a **hard gate**; efficiency only ranks questions that cleared it.
Otherwise the metric is trivially gamed by returning nothing.

## Isolation

Each eval **batch** runs against its own Memgraph database, reset before
fixtures load. This is a reproducibility requirement before it is a hygiene one:
comparing two schema versions is meaningless if the graph also holds whatever
ambient sessions happened to land that week.

Markers inside a batch are provenance only — never the mechanism keeping
eval-agent traces out of the graph under test.

## Development

```bash
uv sync
uv run pytest context-graph/eval/tests
```
