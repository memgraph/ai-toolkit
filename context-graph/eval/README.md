# Context Graph Eval

The evaluation loop for the [Context Graph](../) family: it measures whether
raw agent activity emerges into memory that can be recalled usefully.

Design decisions live on
[Map: Context-graph emergence pipeline](https://github.com/memgraph/ai-toolkit/issues/297);
this README only summarises what the code does.

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
