# Opt-in exchange recall evaluation

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
configuration. The default remains turn/fact output. The token budget covers
source rows, memory notes and exchange headers, excluding `render()` reading
instructions. It applies independently to each searched space.

## Full BEAM 100K result

The separate
[`beam-100k-recall-exchanges-prototype.json`](baselines/beam-100k-recall-exchanges-prototype.json)
records all **400 complete official verdicts**, all 20 chats, 40 questions per
ability, using upstream revision `b2da22eac88bb0874c64665f13457eb99835774a`.
The graph was reused read-only. Query embedding inference is included in
timed recall; model/type-cache warmup, graph construction, answering and
judging are excluded.

| Metric | Published historical configuration | Exchange prototype |
|---|---:|---:|
| Official BEAM score | 63.60% | 65.46% |
| Mean context tokens | 9,450.7 | 5,865.7 |
| Mean cited evidence coverage | 0.599 | 0.703 |
| Complete official verdicts | 400 | 400 |

The point score increases **1.86 percentage points**, with **37.9% fewer
context tokens**. Per-question grades show 88 gains, 79 losses and 233 ties.
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
tokens. Exchange packing adds work; its measured benefit is the quality/token
tradeoff. Abstention, ordering and multi-session regressions keep it opt-in.

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

Supplemental discovery is bounded to four owned sessions. Above 10,000 global
Action documents it uses at most 512 session-local documents and 2 MiB of
ranking features, retaining the primary pool if that cap is exceeded. Prefixes
are ranking features only; displayed source text is retained in full. Exact
entity search remains linear in owned vectors. The global text count is
request-local, not a transactional snapshot across concurrent writes.
