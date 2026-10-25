# Context Graph Eval

The evaluation loop for the [Context Graph](../) family: it measures whether
raw agent activity emerges into memory that can be recalled usefully.

Design decisions live on
[Map: Context-graph emergence pipeline](https://github.com/memgraph/ai-toolkit/issues/297);
this README only summarises what the code does.

## Official numbers

The current read path (hybrid recall over a GLiNER2 graph, no LLM in the build
or the recall), measured end to end on two public benchmarks. Each row is a
committed baseline with its settings; the sections below it have the per-type
breakdown and the caveats.

| Benchmark | Questions | **Answer score** | Evidence recall | Retrieval tokens (mean) | Recall only (mean, p95) | End to end (mean, p95) | Build |
|---|---|---|---|---|---|---|---|
| [LongMemEval-S](#version-point-longmemeval-s-hybrid-recall-over-a-gliner2-graph-2026-10-23), full haystacks | 500 | **87.0%** | 0.92 | 6,465 | 2.85 s, 3.03 s | 10.8 s, 22.2 s | 203 h |
| [BEAM 100K](#version-point-beam-100k-hybrid-recall-over-a-gliner2-graph-2026-10-14) | 400 | **63.6%** | 0.60 | 9,451 | 8.8 s, 10.4 s | 31.8 s, 66.6 s | 478 min |

**Answer score** is the headline both share: the share of the answer key the
system's answer gets right, as decided by the benchmark's own judge. On
LongMemEval-S each question is right or wrong (435 of 500, LongMemEval's
gpt-4o judge). On BEAM each question earns partial credit per rubric item
(1, 0.5 or 0) and the score is the mean over its ten abilities, 0.636 (BEAM's
gpt-4.1-mini judge). The two are the same idea on different judges and
granularity, so compare them as "how much of the answer is right", not as a
ranking between benchmarks. **Evidence recall** (the share of the cited
evidence turns that retrieval returned), **retrieval tokens**, recall-only and
end-to-end latency are measured the same way on both.

LongMemEval-S also reports Contextual Recall (retrieval found what the answer
needs) on 332 of 458 questions (72%), abstention 22/30, F1 0.117, BLEU 0.067 and
0 unscored questions. Both ran on an AMD Ryzen 5 2600 (6 cores, 16 GB, no GPU) against
Memgraph 3.13.2.

## Two tiers, scored separately

| | Tier 1 — adopted | Tier 2 — authored |
|---|---|---|
| Source | LongMemEval v1 (`s` / `m`), converted; BEAM | written for this project (the gold slice) |
| Asks | does recall work *mechanically*? | does it work for *what we are building*? |

Tiers are never blended into one headline: an organizational-recall
regression must not hide behind a personal-memory gain (#303).

## Corpus

LongMemEval questions are deepeval `Golden` records in committed JSONL
(`corpus/tier1-longmemeval.jsonl`), deliberately not in Memgraph: the answer
key must not live in the graph the eval mutates, and git proves two compared
runs asked the same questions (#302). Upstream data is fetched from a pinned
revision and cached under `$XDG_CACHE_HOME/context-graph-eval`; only converted
output is committed.

```bash
uv run --package context-graph-eval context-graph-eval build-corpus \
    --limit 100 --out context-graph/eval/corpus/tier1-longmemeval.jsonl
```

`--limit` samples deterministically, stratified by `(question_type,
abstention)`, proportional to upstream with a floor of 2 per stratum, then
shuffled with a fixed seed so that `run --limit N` (a prefix of the committed
corpus) is an unbiased sample too. Bumping `--revision` invalidates prior
baselines. The `oracle` variant is refused: with no distractors, retrieval
scores well by construction.

## Running a batch

```bash
# --schema-info-enabled feeds the graph-agent's schema description
docker run -d --name ai-toolkit-eval-memgraph -p 7689:7687 \
    memgraph/memgraph-mage:latest --schema-info-enabled=true

uv run --package context-graph-eval context-graph-eval run \
    --limit 100 --retrieval-strategy hybrid \
    --judge-model anthropic:claude-sonnet-4-5-20250929
```

The eval instance is **wiped before each batch** — never point
`--memgraph-url` at a shared or development database.

The runner owns the pipeline; deepeval owns scoring underneath:

```
runner  ->  inject -> reconcile -> retrieve  ->  judge  ->  metrics
```

- **Inject** — each question's haystack becomes its own user's sessions.
  Session ids are namespaced by question, since upstream reuses distractors
  across questions and reconciliation resolves one user per session.
- **Reconcile** — the same pass a real session gets (`sessions-graph`).
  `--extraction-backend` picks `gliner2` (default, local, LLM-free) or
  `lightrag` (needs an LLM key). `--ontology` (GLiNER2 only for the last two)
  picks what GLiNER2 extracts against: `fixed` (default,
  `ontologies/longmemeval.yaml`), `default` (hygm's default model), or
  `learned` (the default model, then each user's own model derived and
  adopted via `sessions-graph derive`, `--derive-workers` users at a time,
  before any question is asked; map #431).
- **Retrieve** — `--retrieval-strategy`:
  - `hybrid` — `sessions_graph.recall`, the product read path: turns by
    vector and text search plus the typed graph's facts. `--hybrid-lanes`
    drops lanes (e.g. `turns,text` is the graph ablation).
  - `graph-agent` (CLI default) — an agent writes read-only Cypher against the
    reconciled graph (#300).
  - `text-search` — Memgraph's full-text index over raw turns; never
    reconciles.

  Every strategy answers through the same `answer_prompt`, so a quality
  difference is down to what was retrieved.

`--skip-reconcile` reuses the graph as-is and refuses unless it holds this
run's sessions, reconciled, by the claimed `--extraction-backend`.
`--max-sessions-per-question` trims distractors (evidence is always kept);
such a score is an upper bound. The runner refuses to start if
`CONFIDENT_API_KEY` is set, since deepeval would upload the run.

`--judge-model`/`--agent-model` take `provider:model_id`; a bare id keeps the
role's default provider. A shared judge/agent provider is warned about, since
#304 keeps them apart. LLM credentials resolve from context-graph's config
file (ADR 0002) before the environment.

## Scoring

Quality is judged, cost is counted (#304).

- **Coverage** — decided by LongMemEval's own judge (`official_judge.py`,
  vendored verbatim, `gpt-4o-2024-08-06`) whenever judging runs and an OpenAI
  key is present.
  The deepeval rubrics (Coverage, Preference, Abstention `GEval`s) are reported
  beside it, and decide only when the official judge is off. A question
  answered from an empty retrieval cannot pass, except an abstention.
- **Contextual Recall** — reported, never gating: a correct computed answer
  ("17 days") has no single supporting row.
- **Evidence recall** — share of the question's evidence turns found in the
  retrieval payload; no judge.
- **Efficiency** — tokens in the retrieval payload (`cl100k_base`, pinned).
  Ranks only questions that cleared coverage (#309).
- **BLEU-1, token F1, latency** — judge-free, over every question.

A question the judge could not score is reported `unscored`, never as 0%.

## Comparing runs

```bash
context-graph-eval run --limit 100 --save runs/baseline.json --label baseline
context-graph-eval run --limit 100 --save runs/candidate.json --label candidate --changed "..."
context-graph-eval compare runs/baseline.json runs/candidate.json --noise-floor 4
```

`compare` **refuses** runs measured differently: corpus revision or variant,
judge, agent model, extraction backend, tokenizer, coverage gate, official
judge, or question count. Retrieval strategy is the one axis it does not pin.
Without `--noise-floor` it calls nothing real. A real coverage regression
decides the verdict even when efficiency improved.

`calibrate` derives the noise floor from repeated runs of the same questions,
and reports pass-set stability and per-question pass rates beside it: a
steady rate can hide a different passing set each run.

## BEAM

The [exchange recall evaluation](recall-exchanges.md) records
a separate 400-question prototype baseline for its named configuration over the
retained GLiNER2 graph. Its development used BEAM questions included in the full
run; independent complete-history validation remains outstanding. The published
version points below keep their original configurations and scores. Exchanges
are now the recall default; new hybrid runs use exchange packing unless the
caller explicitly supplies `RecallConfig(packing="turns")`.

`context-graph-eval beam` runs [BEAM](https://github.com/mohammadtavakoli78/BEAM)'s
probing questions through the hybrid read path, judged by BEAM's own judge
(`beam_judge.py`, vendored verbatim: `gpt-4.1-mini`, temperature 0).

```bash
context-graph-eval beam --size 100K --chats 1-20 --limit 100 \
    --extraction-backend gliner2 --save runs/beam-100k-q100.json
```

- One chat is one user; each time-anchored batch becomes a dated session.
- BEAM data is CC BY-SA, so nothing converted is committed; the pinned
  upstream commit keeps runs comparable. `chat_trunecated.json` is preferred,
  as upstream's answering does.
- Each rubric item scores 1, 0.5 or 0; event ordering is scored by normalised
  Kendall tau. The headline is the mean over abilities. The judge is noisy
  and never sees the question; read per-question verdicts before trusting a
  small delta.
- `--limit` asks a subset balanced across abilities and chats; every chat is
  still loaded, so `--skip-reconcile` can reuse the graph.
- `--ontology` and `--hybrid-lanes` work as on `run`.
- A saved run also records end-to-end and recall-only latency, the build's
  wall-clock, the commit, the Memgraph version and the GLiNER2 settings.

### Version point: BEAM 100K, hybrid recall over a GLiNER2 graph (2026-10-14)

The full BEAM 100K benchmark through the shipped read path: all 20 chats and
400 questions, the graph built by GLiNER2 (no LLM in the build or the recall),
answered by gpt-4.1 and scored by BEAM's own judge.
`baselines/beam-100k-gliner2.json` holds the settings, the report and each
question's scores and latencies, but no BEAM text.

```bash
context-graph-eval beam --size 100K --chats 1-20 --extraction-backend gliner2 \
    --ontology default --agent-model openai:gpt-4.1 \
    --memgraph-url bolt://localhost:<port> --save runs/beam-100k-gliner2.json
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

- **Setup:** commit `d585cea`, Memgraph 3.13.2
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
  scored 0.628 (`rerun` in the JSON); the headline is a second answering pass
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

## Version point: LongMemEval-S, hybrid recall over a GLiNER2 graph (2026-10-23)

The full LongMemEval-S benchmark through the shipped read path: all 500
questions with their whole haystacks (~50 sessions each, no
`--max-sessions-per-question` cap), the graph built by GLiNER2 under hygm's
default model (no LLM in the build or the recall, no episode summaries),
answered by gpt-4.1 and judged by LongMemEval's own judge (gpt-4o-2024-08-06).
`baselines/lme-s-hybrid-gliner2.json` holds the settings, the report and each
question's scores and latencies, but no LongMemEval text; the questions are
`corpus/longmemeval-s-500.jsonl` (upstream revision `98d7416c24c7`).

```bash
context-graph-eval build-corpus --variant s --out corpus/longmemeval-s-500.jsonl
context-graph-eval run --variant s --corpus <batch>.jsonl --limit 20 \
    --extraction-backend gliner2 --ontology default --retrieval-strategy hybrid \
    --agent-model openai:gpt-4.1 --memgraph-url bolt://localhost:<port> --save <batch>.json
```

One graph for 500 questions does not fit in memory (every question has its own
~50 sessions and the runner builds all of them), so the run was 25 batches of 20
questions (the corpus shuffled with `random.Random(0)`), the eval Memgraph
wiped before each, and the saved batches merged. A batch is independent of the
others: each question is its own user.

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
- **Setup:** Memgraph 3.13.2 (`memgraph/memgraph-mage@sha256:e28eea42…`, snapshots
  and WAL off), `fastino/gliner2.5-base-v1` with the greedy decoder, every
  recall lane, an AMD Ryzen 5 2600 (6 cores, 16 GB, no GPU). Batches 0-1 ran on the
  code of #552 (before its rebase onto main); the rest add a retry of recall's
  setup on a transient Memgraph error, the only other change.
- **Recall timing:** measured before #548 moved recall into a worker thread, so
  recall ran on the event loop one question at a time and each recall time is
  uncontended. With #548, concurrent questions overlap their recalls: a run
  finishes sooner, but each recall can take longer than reported here.
- **Build and answer:** 203 hours of wall clock across the 25 batches, ~8.1 h per
  20 questions (~1,000 sessions), most of it GLiNER2 on the CPU. One further
  attempt of batch 0 (7.7 h) was lost to that transient error right after its build.
- **One question is a retrieval failure, scored 0:** `0bc8ad93` ("... Did I visit
  with a friend or not?"). Recall's text lane hands the question's trailing
  "not" to Tantivy as its `NOT` operator, the query is a syntax error, and the
  text-only fallback uses the same lane, so no answer is produced. It
  reproduces on a rebuilt graph, so it is a bug in recall and not noise; it
  costs 0.2 pp and is left in the score.
- **The deepeval answer rubrics,** which are reported beside the official judge
  and do not decide coverage, pass 434/500.
- **Not comparable to the 100-question baseline** (`corpus/tier1-longmemeval.jsonl`,
  5-session cap, 88/100): that run predates recall's memories lane (#500), caps
  the haystack, and is a different sample.
- **Where it loses:** multi-session questions (75.9%), which need evidence from
  several sessions at once, then temporal reasoning (85.7%).
- **"N failed"** in each batch's reconcile line counts sessions with
  relationships outside the default model's declared domain/range; every
  session was reconciled.

## Knowledge-graph construction (`kg`)

`context-graph-eval kg` measures reconcile's default graph construction on
BEAM's long multi-domain chats. It needs no Memgraph and no LLM, and takes a
few minutes on a laptop CPU:

```bash
context-graph-eval kg --observe --baseline context-graph/eval/baselines/kg-beam-100k.json
```

It runs the extraction `sessions-graph reconcile` runs by default (GLiNER2
under hygm's default model, its value-only pass and the user resolver) over a
fixed, evenly spread sample of windows (`--windows`, 200) from BEAM 100K chats
1–20 at the pinned revision. It reports:

- **Speed:** seconds per window, plus with `--observe` the open-endpoint pass
  `sessions-graph derive` pays.
- **Volume:** mentions and relations per window, and their distributions by type.
- **Fit:** catch-all share (`Topic`/`Artifact`), value-type share,
  User-headed relations, and coverage (user turns with at least one typed
  relation, the adoption gate's number).
- **Probes:** known facts (the 25:50 personal best, the MoMA visit) from the
  cached LongMemEval corpus when present, else `None`.
- **Stability:** with `--baseline`, how many of the baseline's mentions and
  relations this run reproduces.

`baselines/kg-beam-100k.json` is committed. It holds only metrics and
fingerprints, never BEAM's text, so the licence point above still holds.
**A PR that changes how the graph is built by default** (the GLiNER2
checkpoint, decoder or caps, the default model, the value pass, the resolver,
windowing) reruns this against the baseline and pastes the comparison into the
PR, saving a new baseline in the same PR if the change is intended. Seconds
per window depend on the machine; compare them only against a baseline from
the same hardware.

## The gold slice

Tier 1 injects fixtures directly, so it never exercises the capture layer.
The gold slice drives a **real, billed Claude Code session** with hooks live
and scores what was actually recorded — the only eval coverage for subagent
nesting (#308). Its one question asks for a fact that exists only inside a
subagent.

```bash
context-graph-eval gold-slice --memgraph-url bolt://localhost:7689
context-graph-eval run --gold-slice --skip-reconcile --memgraph-url bolt://localhost:7689
```

The planting prompt must not contain the fact, the fact must be unguessable
(a hex revision), and the prompt must contain no literal `SKILL.md` path
(#293). The run refuses unless the fact is planted, nested under an `Agent`,
and absent from top level. Hooks are pointed at the eval instance through a
throwaway config named by `CONTEXT_GRAPH_CONFIG` (ADR 0003).

## Known limitations

- **Reconciliation dominates cost.** Eval batches raise LightRAG's
  merge-summary threshold, match its LLM concurrency knobs, serialize local
  embedding calls, and embed with `BAAI/bge-m3` (8192-token chunks) — the same
  code path as production with different settings (`reconcile.py`).
- **The noise floor is large relative to deltas**; calibrate on the current
  corpus size before trusting a small coverage change.
- **Tier 2 has one question.**

## Development

```bash
uv sync
uv run --package context-graph-eval --extra test --extra pipeline pytest context-graph/eval/tests
```

Tests read `EVAL_MEMGRAPH_URL` (default `bolt://localhost:7689`) and skip when
no instance is reachable.
