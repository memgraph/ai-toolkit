# Recall benchmark results

[Back to the evaluation guide](../README.md)

Paths in commands and file references below are relative to `context-graph/eval/`.

Each version point is named for the memgraph-context-graph release whose code
it measured; the components it ships with are listed under **Setup**.

## memgraph-context-graph 0.12.0: LongMemEval-S (2026-10-28)

The full LongMemEval-S benchmark through the shipped read path as it is now:
recall's default exchange packing (#610, #611) over the same kind of graph as
[0.6.0](#memgraph-context-graph-060-longmemeval-s-2026-10-23)
(GLiNER2 under hygm's default model, no LLM in the build or the recall). All
500 questions with their whole haystacks, answered by gpt-4.1 and judged by
LongMemEval's own judge (gpt-4o-2024-08-06).
`baselines/lme-s-exchanges-gliner2.json` holds the settings, the report and
each question's scores and latencies, but no LongMemEval text.

```bash
context-graph-eval run lme-s
```

| Question type | Questions | Accuracy | Evidence recall |
|---|---|---|---|
| single-session-user | 70 | 100.0% | 1.00 |
| single-session-assistant | 56 | 94.6% | 1.00 |
| knowledge-update | 78 | 94.9% | 1.00 |
| single-session-preference | 30 | 80.0% | 0.94 |
| temporal-reasoning | 133 | 86.5% | 0.92 |
| multi-session | 133 | 75.9% | 0.93 |
| **All** | **500** | **87.4% (437/500)** | **0.96** |

Abstention (30 of the 500, inside the types above): 23 correct.

| | Mean | Median | p95 |
|---|---|---|---|
| End to end (recall, answer) | 2.92 s | 1.98 s | 6.66 s |
| Recall only (`sessions_graph.recall`, no answering model) | 1.56 s | 0.65 s | 5.13 s |

Retrieval returns 5,863 tokens on average (median 5,888 over the passed
questions), against 6,465 for turn packing. F1 is 0.118 and BLEU 0.068.

- **Against 0.6.0:** the score is level (+0.4 pp, inside
  the ~3-point drift between runs), evidence recall rises from 0.92 to 0.96,
  and retrieval is 9% smaller. Per type the moves are within noise except
  single-session-preference (90.0% to 80.0%, 3 of 30 questions).
- **Misses:** 56 outside abstention. On 28 of them recall returned all of the
  cited evidence and on 21 some of it, so most misses are in answering, not
  retrieval; on 7 it returned none.
- **`0bc8ad93` now answers:** the trailing-"not" query that the text lane once
  handed to Tantivy as a syntax error retrieves normally and scores.
- **Setup:** memgraph-context-graph 0.12.0 with sessions-graph 0.12.0 (commit
  `e31e945`), Memgraph 3.13.2
  (`memgraph/memgraph-mage@sha256:e28eea42…`, write-ahead log and snapshots on,
  as the eval instances now run), `fastino/gliner2.5-base-v1` with the greedy
  decoder, every recall lane, an Apple M2 Pro (10 cores, 16 GB, Docker limited
  to 8 GB, no GPU). Four questions are answered at a time.
- **Build:** 25 shards of 20 questions as `run lme-s` builds them (same
  workload, shard order and build fingerprint), each kept in its own instance
  instead of wiped, then asked. ~41 h of build in all, ~1.7 h per shard
  (~950 sessions); every session reconciled.
- **Recall timing** is measured with recall in a worker thread (#548) and four
  questions in flight, so recalls overlap and the tail is longer than the
  median.

## memgraph-context-graph 0.12.0: BEAM 100K (2026-10-28)

The full BEAM 100K benchmark through the shipped read path as it is now:
recall's default exchange packing over a freshly rebuilt GLiNER2 graph, all 20
chats and 400 questions, answered by gpt-4.1 and scored by BEAM's own judge
(gpt-4.1-mini). `baselines/beam-100k-exchanges-gliner2.json` holds the
settings, the report and each question's scores and latencies, but no BEAM
text.

```bash
context-graph-eval run beam-100k --rebuild
```

| Ability | Score | Evidence recall |
|---|---|---|
| preference_following | 0.919 | 0.76 |
| information_extraction | 0.779 | 0.82 |
| instruction_following | 0.756 | 0.53 |
| knowledge_update | 0.725 | 0.91 |
| multi_session_reasoning | 0.725 | 0.70 |
| contradiction_resolution | 0.672 | 0.92 |
| abstention | 0.525 | – |
| summarization | 0.471 | 0.32 |
| temporal_reasoning | 0.463 | 0.97 |
| event_ordering | 0.278 | 0.34 |
| **BEAM score (mean of abilities)** | **0.631** | |

| | Mean | Median | p95 |
|---|---|---|---|
| End to end (recall, answer) | 2.50 s | 2.22 s | 4.01 s |
| Recall only (`sessions_graph.recall`, no answering model) | 0.50 s | 0.48 s | 0.63 s |

Retrieval returns 5,865 tokens on average (median 5,888 over the passed
questions), against 9,451 for turn packing.

- **Against [0.6.0](#memgraph-context-graph-060-beam-100k-2026-10-14):** the score is level (0.631 against
  0.636, inside BEAM's ~0.04 drift between runs), evidence recall rises from
  0.60 to 0.70, retrieval is 38% smaller, and recall is ~17x faster (0.50 s
  against 8.8 s mean). Abilities moved by up to 0.10 either way
  (multi-session reasoning down, knowledge update up), at the ~0.08 noise
  floor of a 40-question ability.
- **Setup:** memgraph-context-graph 0.12.0 with sessions-graph 0.12.0 (commit
  `e31e945`; the run records `e31e945-dirty` for an
  untracked driver script in the checkout; no module that builds or reads the
  graph differed), Memgraph 3.13.2 (`memgraph/memgraph-mage@sha256:e28eea42…`),
  `fastino/gliner2.5-base-v1` with the greedy decoder, hygm's default model,
  every recall lane, BEAM revision `b2da22ea`, an Apple M2 Pro (10 cores,
  16 GB, Docker limited to 8 GB, no GPU). Four questions are answered at a time.
- **Build:** 120 min for the 82 sessions, every one reconciled.
- **Every question scored:** no answering timeouts and none unscored.
- **Where it loses:** event ordering, temporal reasoning and summarization, as
  before. Temporal reasoning finds 97% of the cited evidence, so its misses
  are in answering from it.

## memgraph-context-graph 0.6.0: LongMemEval-S (2026-10-23)

The full LongMemEval-S benchmark through the shipped read path: all 500
questions with their whole haystacks (~50 sessions each, no
`--max-sessions-per-question` cap), the graph built by GLiNER2 under hygm's
default model (no LLM in the build or the recall, no episode summaries),
answered by gpt-4.1 and judged by LongMemEval's own judge (gpt-4o-2024-08-06).
`baselines/lme-s-hybrid-gliner2.json` holds the settings, the report and each
question's scores and latencies, but no LongMemEval text; the questions are
`corpus/longmemeval-s-500.jsonl` (upstream revision `98d7416c24c7`).

```bash
context-graph-eval run lme-s --packing turns
```

One graph for 500 questions does not fit in memory (every question has its own
~50 sessions), so the run was 25 shards of 20 questions (the corpus shuffled
with `random.Random(0)`), the eval Memgraph wiped before each, and the saved
shards merged -- what `run lme-s` now does on its own.

| Question type | Questions | Accuracy | Evidence recall |
|---|---|---|---|
| single-session-user | 70 | 97.1% | 0.98 |
| single-session-assistant | 56 | 96.4% | 1.00 |
| knowledge-update | 78 | 91.0% | 0.97 |
| single-session-preference | 30 | 90.0% | 0.89 |
| temporal-reasoning | 133 | 85.7% | 0.87 |
| multi-session | 133 | 75.9% | 0.89 |
| **All** | **500** | **87.0% (435/500)** | **0.92** |

Abstention (30 of the 500, inside the types above): 22 correct.

| | Mean | Median | p95 |
|---|---|---|---|
| End to end (recall, answer) | 10.8 s | 10.0 s | 22.2 s |
| Recall only (`sessions_graph.recall`, no answering model) | 2.85 s | 2.86 s | 3.03 s |

Retrieval returns 6,465 tokens on average (median 6,414). F1 is 0.117 and BLEU
0.067: the answerer writes full sentences against short reference answers, so
read accuracy, not those two.

- **Other metrics:** Contextual Recall passed (at 0.7) on 332 of 458 judged
  questions; the median retrieval payload over the 435 covered questions is
  6,378 tokens; 0 questions unscored (the judge answered every one).
- **Setup:** memgraph-context-graph 0.6.0 with sessions-graph 0.8.1 (branched
  from main at #546), Memgraph 3.13.2
  (`memgraph/memgraph-mage@sha256:e28eea42…`, snapshots
  and WAL off), `fastino/gliner2.5-base-v1` with the greedy decoder, every
  recall lane, an AMD Ryzen 5 2600 (6 cores, 16 GB, no GPU). Shards 0-1 ran on the
  code of #552 (before its rebase onto main); the rest add a retry of recall's
  setup on a transient Memgraph error, the only other change.
- **Recall timing:** measured before #548 moved recall into a worker thread, so
  recall ran on the event loop one question at a time and each recall time is
  uncontended. With #548, concurrent questions overlap their recalls: a run
  finishes sooner, but each recall can take longer than reported here.
- **Build and answer:** 203 hours of wall clock across the 25 shards, ~8.1 h per
  20 questions (~1,000 sessions), most of it GLiNER2 on the CPU. One further
  attempt of shard 0 (7.7 h) was lost to that transient error right after its build.
- **One question is a retrieval failure, scored 0:** `0bc8ad93` ("... Did I visit
  with a friend or not?"). Recall's text lane hands the question's trailing
  "not" to Tantivy as its `NOT` operator, the query is a syntax error, and the
  text-only fallback uses the same lane, so no answer is produced. It
  reproduces on a rebuilt graph, so it is a bug in recall and not noise; it
  costs 0.2 pp and is left in the score.
- **The deepeval answer rubrics,** which are reported beside the official judge
  and do not decide coverage, pass 434/500.
- **Not comparable to the earlier 100-question baseline** (5-session cap,
  88/100): that run predates recall's memories lane (#500), capped the
  haystack, and was a different sample.
- **Where it loses:** multi-session questions (75.9%), which need evidence from
  several sessions at once, then temporal reasoning (85.7%).
- **"N failed"** in each shard's reconcile line counts sessions with
  relationships outside the default model's declared domain/range; every
  session was reconciled.

## memgraph-context-graph 0.6.0: BEAM 100K (2026-10-14)

The full BEAM 100K benchmark through the shipped read path: all 20 chats and
400 questions, the graph built by GLiNER2 (no LLM in the build or the recall),
answered by gpt-4.1 and scored by BEAM's own judge.
`baselines/beam-100k-gliner2.json` holds the settings, the report and each
question's scores and latencies, but no BEAM text.

```bash
context-graph-eval run beam-100k --packing turns
```

| Ability | Score | Evidence recall |
|---|---|---|
| preference_following | 0.906 | 0.59 |
| multi_session_reasoning | 0.829 | 0.56 |
| information_extraction | 0.778 | 0.73 |
| instruction_following | 0.700 | 0.25 |
| contradiction_resolution | 0.675 | 0.89 |
| knowledge_update | 0.625 | 0.85 |
| abstention | 0.600 | – |
| temporal_reasoning | 0.475 | 0.94 |
| summarization | 0.468 | 0.26 |
| event_ordering | 0.303 | 0.28 |
| **BEAM score (mean of abilities)** | **0.636** | |

| | Mean | Median | p95 |
|---|---|---|---|
| End to end (recall, answer) | 31.8 s | 28.2 s | 66.6 s |
| Recall only (`sessions_graph.recall`, no answering model) | 8.8 s | 8.9 s | 10.4 s |

- **Setup:** memgraph-context-graph 0.6.0 with sessions-graph 0.8.1 (commit
  `d585cea`), Memgraph 3.13.2
  (`memgraph/memgraph-mage@sha256:e28eea42…`), `fastino/gliner2.5-base-v1`
  with the greedy decoder, hygm's default model, every recall lane, BEAM
  revision `b2da22ea`. Four questions are answered at a time.
- **Recall timing:** measured before #548 moved recall into a worker thread, so
  recall ran on the event loop one question at a time and each recall time is
  uncontended. With #548, concurrent questions overlap their recalls: the run
  finishes sooner, but each recall can take longer than reported here.
- **Build:** 478 min for the 82 sessions (1.59M words) on an AMD Ryzen 5 2600
  (6 cores, 16 GB, no GPU). Most of it is GLiNER2 on the CPU, then
  Memgraph embedding the passages, entities and facts.
- **Noise:** the run that built the graph also answered and judged it, and
  scored 0.628 (`notes.first_pass` in the JSON); the headline is a second answering pass
  over the same graph. Abilities moved by up to 0.075 between the two (40
  questions each); retrieval, and so evidence recall, was identical. Read an
  ability's delta under ~0.08 as noise.
- **One question is a timeout, scored 0:** the answering call to OpenAI timed
  out on `beam-100K-8-summarization-1` (on two other questions in the first
  pass). That costs the headline about 0.003.
- **Recall is slow on long histories.** A BEAM user has ~250 long turns, and
  recall scores every one of that user's passages, entities and facts by
  brute-force cosine. One chat alone recalls in ~4.8 s; with all 20 loaded
  and four questions at once it is 8.8 s, against ~1 s on LongMemEval.
- **Where it loses:** event ordering, summarization and temporal reasoning.
  Temporal reasoning finds 94% of the cited evidence, so its misses are in
  answering from it, not in recall.
- **Truncation:** reconciliation extracts at most 8,000 characters of a
  message (`MAX_RECONCILABLE_CHARS`). 20 of BEAM's 5,094 messages are longer,
  3.9% of its text, nearly all of it one 349k-character paste in chat 4;
  those messages are still embedded in full for the turn lanes.
- **"82 failed"** in the reconcile line counts sessions with relationships
  outside the default model's declared domain/range; every session was
  reconciled.
