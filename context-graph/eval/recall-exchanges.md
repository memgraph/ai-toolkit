# Experimental exchange recall evaluation

The recall implementation batches source lookups, groups shared chunks before
fact/entity expansion, preserves deterministic cutoffs and fills passage-hit
pools with distinct sources. Natural-language text tokens, including trailing
`not`, are searched literally. These changes run against the existing graph.

The optional exchange mode uses graph links to keep related user/assistant
messages together, ranks source groups with reciprocal rank fusion, and packs
whole groups inside a token budget. Extracted facts vote for source messages
without becoming separate answer-context rows. It uses the existing local
GLiNER2 graph and BGE-small vectors; no graph reconstruction or larger model
is required.

```python
from sessions_graph import RecallConfig
from sessions_graph.recall import recall

config = RecallConfig(
    packing="exchanges",
    token_budget=6000,
    supplemental_search=True,
)
result = recall(db, user_id, question, config=config)
```

Install `sessions-graph[compact]`. The same options work in the hub's `[recall]`
configuration. The default retains turn/fact formatting, but retrieval behavior
changes: distinct-source filling replaces a passage-only cutoff, text words are
literal, and fact selection follows entity priority with deterministic ties.
Keeping the formatting default does not preserve historical retrieval output.
The token budget covers source rows, memory notes and exchange headers, excluding `render()` reading
instructions. It applies independently to each searched space.

## Status and generalization

Exchange packing and supplemental discovery are experimental, opt-in policies.
The 65.46% result applies to the exact configuration above; the baseline remains
marked `prototype`. Default recall uses `packing="turns"` and
`supplemental_search=False`. The published historical version points retain
their original settings and scores. No complete 400-question score has been
established for the optimized default alone.

Batching and source-ownership changes apply to both packing modes; their
performance depends on graph shape and indexes. Exchange widths,
rank-fusion weights, the token budget, problem/fix cue vocabulary and search
thresholds were selected through repeated BEAM 100K experiments. The complete
run includes questions used during development, so it is not an independent
holdout. Production retrieval uses the question and owned graph contents;
benchmark question IDs, ability labels and reference answers are not policy
inputs. That separation does not establish generalization.

Before promoting exchanges to a default or claiming broader memory-quality
gains, freeze the policy and validate it on independent, complete histories.
Include answerable controls as well as abstention, ordering and multi-session
cases, and measure tokens and latency under a matched protocol. The reduced
LME checks below are diagnostic and do not satisfy this gate. Further tuning
on the same BEAM failures must remain development evidence.

## Matched default validation, 2026-10-25

The metric-only
[`beam-100k-recall-review-diagnostic.json`](baselines/beam-100k-recall-review-diagnostic.json)
records a new **400-question retrieval comparison** and a separate **20-question
paired answer-quality pilot**. These are different sample sizes. No complete
freshly judged 400-question optimized-default quality score is claimed.

All three retrieval configurations ran sequentially on the same retained graph,
rotating their order per question. Historical recall uses the pre-PR `e9194ce`
implementation; the proposed implementation is committed at `a4870ef`, and all
four production source files exactly match that commit. Full source hashes,
configs and environment versions are recorded in the diagnostic. Query
embedding inference is timed; model/type initialization, answering and judging
are excluded. No graph reconstruction or writes occurred.

| Metric, all 400 questions | Historical default | Proposed default | Exchanges |
|---|---:|---:|---:|
| Mean context tokens | 9,472.1 | 9,514.8 | 5,865.7 |
| Mean recall | 8.961 s | 0.744 s | 0.859 s |
| p95 recall | 10.337 s | 0.881 s | 1.018 s |
| Mean cited evidence coverage | 0.5993 | 0.6265 | 0.7029 |
| Retrieval errors | 0 | 0 | 0 |

The proposed default changes rendered context on **375/400 questions**, with
**0.45% more mean context tokens**. It does not preserve historical output.
Exchanges use **38.1% fewer tokens** than the contemporaneous historical
configuration, and all **400 exchange contexts exactly match** those from the
frozen 65.46% prototype run. Evidence coverage is a retrieval diagnostic, not
an answer-quality score. Source ownership and active vector lanes were checked
for every request; the retained node count stayed unchanged.

The retained graph lacks the Action ID property index that normal setup
creates. This comparison does not establish the same speedup on indexed
production graphs. The synthetic lookup table below measures that distinction.
Short local validation/scaling tasks overlapped the initial portion of this
run; it was not a perfectly idle-host microbenchmark.

The quality pilot selected two questions per ability across all 20 chats using
a fixed hash ordering, before retrieval results were inspected. Both defaults
used `gpt-4.1` at temperature zero and the unchanged official `gpt-4.1-mini`
rubric, including ordering alignment. A fresh run-local full-request cache was
shared across the paired configurations; older answers/judgments were not
reused. No policy changes followed inspection of the pilot results.

| Quality pilot only | Historical default | Proposed default |
|---|---:|---:|
| Complete judgments | 20 | 20 |
| Mean sample score | 59.01% | 66.76% |

There were **4 gains, 2 losses and 14 ties**, with no unjudged cases or request
errors. The exploratory paired bootstrap 95% interval for the score difference
is **-4.5 to +21.5 percentage points**. This small panel neither establishes a
quality improvement nor rules out a regression; it is not a new official
benchmark score or an independent holdout. Only one answer/judge generation
was made per distinct request, so model/judge variance is not estimated.

The pilot used 254 new API requests, below its 600-attempt ceiling: 460,378
prompt tokens and 16,192 completion tokens. Paid judging was limited to these
20 selected questions; all 400-question comparisons above are retrieval-only.
Published official numbers remain unchanged.

## Earlier full exchange prototype result

The separate
[`beam-100k-recall-exchanges-prototype.json`](baselines/beam-100k-recall-exchanges-prototype.json)
records all **400 complete official verdicts**, all 20 chats, 40 questions per
ability, using upstream revision `b2da22eac88bb0874c64665f13457eb99835774a`.
The graph was reused read-only. Query embedding inference is included in
timed recall; model/type-cache warmup, graph construction, answering and
judging are excluded.

| Metric | Published historical configuration | Exchange prototype |
|---|---:|---:|
| Mean context tokens | 9,450.7 | 5,865.7 |
| BEAM score (official judge) | 63.60% | 65.46% |
| Mean cited evidence coverage | 0.599 | 0.703 |
| Complete official verdicts | 400 | 400 |

**37.9% fewer context tokens** is the observed reduction against the historical
run. The point-score difference is **1.86 percentage points**, but does not
establish a quality improvement under a matched protocol. Per-question grades
show 88 gains, 79 losses and 233 ties.
This compares the combined query/packing changes against the published
historical configuration, not packing alone against a freshly judged
optimized default.

| Ability | Published | Exchange |
|---|---:|---:|
| Abstention | 60.00% | 52.50% |
| Contradiction resolution | 67.50% | 71.88% |
| Event ordering | 30.30% | 28.70% |
| Information extraction | 77.81% | 81.46% |
| Instruction following | 70.00% | 83.13% |
| Knowledge update | 62.50% | 70.00% |
| Multi-session reasoning | 82.93% | 75.93% |
| Preference following | 90.63% | 91.25% |
| Summarization | 46.83% | 49.80% |
| Temporal reasoning | 47.50% | 50.00% |

Sequential warm recall: **0.864 s mean, 0.860 s median, 0.993 s p95**.
All source IDs were owned, all vector lanes ran, there were no empty contexts,
and every context fit 6,000 tokens (maximum 5,984). Historical recall latency
predates threaded recall and used different concurrency, so no matched
speedup over that historical timing is claimed.

A smaller matched eight-question retrieval panel measured 0.786 s for the
optimized default versus 0.863 s with exchanges, with 9,273 versus 5,896 mean
tokens. Exchange packing adds work; this panel measures a token reduction, not
an established answer-quality improvement. Abstention, ordering and multi-session
regressions keep it opt-in.

Answering used the unchanged shared prompt and `gpt-4.1`, temperature zero.
Judging used the official `gpt-4.1-mini` BEAM rubric; ordering retained upstream
equivalence alignment and normalized Kendall tau. Full-request caches and
checkpoints were used, and infrastructure failures remained unjudged until
retried successfully. A single answer/judge generation does not measure
model variance. Staged answer generation and cache reuse do not establish
comparable end-to-end latency.

One timeout case returned a pathological whitespace-heavy answer and received
an incorrect positive official abstention grade. That grade is retained for
comparability and flagged here, not treated as evidence of sound abstention.
Setting only that grade to zero would yield 65.21% for this prototype. No
manual score overrides or selective replacement answers were used.

A later universal source-provenance context annotation scored 64.68% on all
400 questions and is not part of this implementation. Raw benchmark text,
contexts, answers, request caches and experimental scripts are not included
in this PR; the baseline contains settings, aggregates and per-question metrics.

## LongMemEval and implementation checks

A genuine GLiNER2/BGE-small partial-history graph was built for two LME
questions, then expanded with eight question-relevant real distractor sessions.
The resulting 15-session, 195-message graph preserved all selected evidence.
Both questions were judged correct with the optimized default and exchanges.
Mean context tokens were 5,903 versus 5,432 (8.0% fewer), with warm recall of
247 versus 279 ms. These evidence-preserving partial histories are diagnostic;
they do not establish full LME accuracy or a latency improvement over the
historical 2.85-second complete benchmark. Published LME scores are unchanged.

Real-Memgraph regressions cover shared chunks, distinct-source limits,
literal text queries, indexed/unindexed source lookup, ownership before fact
confidence cutoffs, direction and session/agent boundaries, foreign-container
neighbors, fallback, token budgets and SDK output. Config/CLI tests verify
normal boolean and optional-value round trips. The frozen Community recall
output remains unchanged. The implementation also reproduced 20 archived
compact/complementary contexts exactly before the full run.

Exact entity search remains linear in owned vectors. The policy limits and
concurrent-write caveats are detailed below.

## Installation, offline use and policy limits

The first `tiktoken.get_encoding("cl100k_base")` downloads encoding data.
Exchange recall initializes it before graph queries or embedding inference and
reports an actionable error if initialization fails. For an offline server,
set `TIKTOKEN_CACHE_DIR` to a persistent directory and run the following with
network access, using the same directory in the offline process:

```sh
python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"
```

Memory notes are packed first. Oversized exchanges are skipped whole. Enabled
vector/text pools widen to at least 24 sources and fact-source turns to 16;
zero widths and disabled lanes remain disabled. Up to three directed
`FOLLOWED_BY` edges join messages within their session or agent. Exchange RRF
uses damping 20, direct-lane weight 1 and graph-lane weight 1/3. These are
experimental constants selected during BEAM 100K development, not universal
retrieval defaults.

Supplemental discovery uses English problem/fix cues in at most four
vector-selected owned sessions and selects up to 24 sources. The primary and
supplemental text lanes each receive weight 0.5, and at most one quarter of the
remaining row budget is initially reserved for newly discovered groups.
At up to 10,000 global Action documents, indexed discovery reuses the main text
lane's corpus count. Above that, or when the count is unavailable, local BM25
examines at most 512 documents, with 8,192-character prefixes and a total
2,097,152-character feature limit (characters, not bytes). Original sources are
fetched separately. Exceeding the cap preserves primary candidates; missing
vector/text lanes skip discovery, and query failures propagate. Concurrent
inserts can change candidates because the count is not a transactional snapshot.

In multi-space recall each space packs independently. Fused `turns` and `facts`
remain available; rendering retains each space's group boundaries and excerpts.
`space_results` holds the original scoped results, serialized under `spaces`,
so identical source IDs from different databases remain distinct. Memory notes
still compete for the existing shared `memories_k` limit; there is no global
fused token ceiling. Mixed packing configurations retain each space's format.

## Fact-source lookup scaling diagnostic

A disposable synthetic graph held one user's 10k or 100k Actions, 15 selected
entities, 1,500 candidate fact edges and 90 selected facts. Two warm samples per
configuration followed one warmup; these timings isolate `_ENTITY_FACTS`, not
full recall or benchmark answer quality.

| Actions | Action ID index | Historical lookup | Batched lookup |
|---|---|---:|---:|
| 10,000 | absent | 31 ms | 236 ms |
| 10,000 | present | 21 ms | 233 ms |
| 100,000 | absent | 86,498 ms | 281 ms |
| 100,000 | present | 25 ms | 278 ms |

Batching avoids the severe unindexed plan at 100k Actions, but is slower for
this indexed, high-candidate-fanout workload. It is not a universal per-query
speedup. The `own` list contains owned candidate source IDs, not every message
in the user's history; its membership checks still cost more as candidate
fanout grows. Normal setup creates the Action ID index. The matched full-recall
comparison below measures the combined implementation separately.
