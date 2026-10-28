# Context Graph Eval

The evaluation loop for the [Context Graph](../) family: it measures whether
raw agent activity emerges into memory that can be recalled usefully.

Design decisions live on
[Map: Context-graph emergence pipeline](https://github.com/memgraph/ai-toolkit/issues/297);
this README only summarises what the code does.

## Official numbers

The current read path (hybrid recall over a GLiNER2 graph, no LLM in the build
or the recall), measured end to end on two public benchmarks: the
[official workloads](#workloads), whole and uncapped. These are the numbers we
communicate. Each row is a committed baseline in `baselines/` with its
settings; the [benchmark reports](docs/benchmark-results.md) have the per-type
breakdowns, reproduction commands, and caveats.

| Benchmark | Questions | Answer score | Evidence recall | Retrieval tokens | Recall latency (mean) |
|---|---|---|---|---|---|
| [LongMemEval-S](docs/benchmark-results.md#version-point-longmemeval-s-exchange-recall-over-a-gliner2-graph-2026-10-28), full haystacks | 500 | **87.4%** | 0.96 | 5,863 | 1.56 s |
| [BEAM 100K](docs/benchmark-results.md#version-point-beam-100k-exchange-recall-over-a-gliner2-graph-2026-10-28) | 400 | **63.1%** | 0.70 | 5,865 | 0.50 s |

**Answer score** is the headline both share: the share of the answer key the
system's answer gets right, as decided by the benchmark's own judge. On
LongMemEval-S each question is right or wrong (437 of 500, LongMemEval's
gpt-4o judge). On BEAM each question earns partial credit per rubric item
(1, 0.5 or 0) and the score is the mean over its ten abilities, 0.631 (BEAM's
gpt-4.1-mini judge). The two are the same idea on different judges and
granularity, so compare them as "how much of the answer is right", not as a
ranking between benchmarks. **Evidence recall** (the share of the cited
evidence turns that retrieval returned), **retrieval tokens**, recall-only and
end-to-end latency are measured the same way on both.

Both were measured with recall's default exchange packing at commit `e31e945`.
Against the previous version points, measured with turn packing (separate turn
and fact rows), the answer scores are level and retrieval finds more of the
evidence in fewer tokens, faster; the
[benchmark reports](docs/benchmark-results.md) keep both. To re-measure the
earlier points as they were, run them with `--packing turns`.

LongMemEval-S also reports abstention 23/30, F1 0.118, BLEU 0.068 and 0
unscored questions; BEAM had none unscored. Both ran on an Apple M2 Pro
(10 cores, 16 GB, no GPU) against Memgraph 3.13.2.

## Workloads

Everything the eval runs is a named **workload**: a fixed set of questions and
the haystack they are asked of. There are two kinds.

- **Official** workloads are whole benchmarks, uncapped. Their results are the
  official numbers above, and they take days.
- **Dev** workloads are small, fixed slices of the same benchmarks, sized for
  iterating on recall and construction. Compare dev runs with each other, never
  with an official number: a capped haystack makes LongMemEval easier, and four
  BEAM chats are not BEAM.

| Workload | Kind | What it asks | Instance |
|---|---|---|---|
| `lme-s` | official | LongMemEval-S, all 500 questions, full haystacks (25 shards of 20) | `:7791` |
| `beam-100k` | official | BEAM 100K, all 20 chats and 400 questions | `:7793` |
| `lme-s-dev` | dev | LongMemEval-S, 50 questions stratified by type, haystacks capped at 10 sessions | `:7792` |
| `beam-100k-dev` | dev | BEAM 100K chats 1, 6, 11 and 16, all 80 of their questions | `:7794` |
| `gold-slice` | dev | Tier 2: one fact that exists only inside a subagent ([below](#the-gold-slice)) | `:7795` |

`context-graph-eval workloads` lists them with the state of each instance and
the graph it keeps. The LongMemEval questions are committed JSONL
(`corpus/longmemeval-s-500.jsonl`, `corpus/longmemeval-s-dev.jsonl`),
deliberately not in Memgraph: the answer key must not live in the graph the
eval builds, and git proves two compared runs asked the same questions (#302).
Upstream data is fetched from a pinned revision and cached under
`$XDG_CACHE_HOME/context-graph-eval`; BEAM is CC BY-SA, so none of it is
committed.

## Running a workload

```bash
uv sync --all-packages --all-extras
uv run context-graph-eval run lme-s-dev
```

One pipeline runs every workload, the same one the product runs:

```
build:  inject -> reconcile -> (derive) -> embed      once per construction
ask:    recall -> answer -> benchmark's judge         every run
```

- **Inject** writes each haystack session under its user: on LongMemEval each
  question is its own user, on BEAM each chat is.
- **Reconcile** extracts with local GLiNER2 under hygm's default model, the
  pass a real session gets. `--ontology learned` then derives and adopts each
  user's own model (`sessions-graph derive`, map #431).
- **Embed** stores recall's vectors in Memgraph.
- **Recall** is `sessions_graph.recall`, the product read path, with every
  lane on and its default packing (exchanges); `--lanes turns,text` is the
  graph ablation and `--packing turns` the separate turn/fact rows the
  version points were measured with. Every benchmark answers
  through the same prompt (`answering.answer_prompt`) with gpt-4.1
  (`--agent-model`), so a score difference is down to what recall found.
- **Judge** is each benchmark's own, vendored verbatim: LongMemEval's
  (gpt-4o-2024-08-06) and BEAM's (gpt-4.1-mini).

A run prints its report and saves itself to `runs/<workload>/<label>.json`
(git-ignored; the label defaults to the commit). It needs an `OPENAI_API_KEY`,
resolved from context-graph's config file (ADR 0002) before the environment,
and refuses to start if `CONFIDENT_API_KEY` is set, since deepeval would
upload the run.

## Eval instances keep their graphs

Building a graph takes hours; asking it takes minutes. So every workload has
its own Memgraph container, `context-graph-eval-<workload>`, created on first
use with its data on a named volume, and a run asks the graph it finds there
when nothing that shapes the graph has changed.

The graph carries a manifest (an `(:EvalBuild)` node) with a fingerprint of
what built it: the workload's questions, the build settings (ontology, GLiNER2
checkpoint, decoder and caps, embedding model) and the source of every module
that shapes the graph (actions-graph, unstructured2graph, hygm, the
construction half of sessions-graph, and eval's own injection). On each run:

| The instance holds | The run |
|---|---|
| a finished build with the same fingerprint | asks it (a recall or answering change: minutes) |
| an interrupted build with the same fingerprint | finishes it: sessions already written and reconciled are kept |
| anything else | wipes it and builds again, saying which input changed |

`--rebuild` forces a rebuild. `--reuse` asks a finished graph of the same
questions even though its inputs changed; the run is marked stale.
`context-graph-eval build <workload>` builds and stops (e.g. overnight), and
`context-graph-eval drop <workload>` deletes the instance and its graph.

The instances are durable: the write-ahead log records every change, a
snapshot is taken on exit, and both are recovered on start, so a crash or a
restart loses nothing. They run `memgraph/memgraph-mage:3.13.2` (3.9.0 broke
recall's text search), listen on loopback only, and are dedicated to eval:
a rebuild wipes them. `--memgraph-url` points a run at an instance managed
elsewhere (CI) instead.

`lme-s` does not fit one graph in 16 GB, so it is asked in 25 shards of 20
questions (the corpus shuffled with `random.Random(0)`), one shard in the
instance at a time. Each shard's result is saved under
`runs/lme-s/<label>.shards/` as it finishes, and a rerun with the same label
skips the shards already done. A shard is independent of the others: each
question is its own user.

## Scoring

Every question gets a score from its benchmark's own judge, between 0 and 1,
and quality is judged while cost is counted (#304):

- **LongMemEval** — 1 when LongMemEval's judge says the answer is right, 0
  when not. `--rubric-judge` adds the deepeval rubrics (Coverage, Preference,
  Abstention `GEval`s and Contextual Recall) beside it as diagnostics; they
  decide only questions no upstream template covers (the gold slice). The
  headline is accuracy.
- **BEAM** — the mean of the question's rubric items at 1, 0.5 or 0, or the
  normalised Kendall tau for event ordering. The headline is the mean over
  abilities, as upstream reports it. The judge never sees the question; read
  per-question verdicts before trusting a small delta.
- **On both** — a question answered from an empty retrieval scores 0, except
  an abstention. Evidence recall (the share of cited evidence turns recall
  returned), retrieval tokens (`cl100k_base`), recall-only and end-to-end
  latency need no judge. A question the judge could not score is unjudged and
  left out of every mean, never counted as 0.

## Comparing runs

```bash
context-graph-eval run lme-s-dev --label before
# ... change recall or construction ...
context-graph-eval run lme-s-dev --label after --changed "what changed"
context-graph-eval compare runs/lme-s-dev/before.json runs/lme-s-dev/after.json --noise-floor 4
```

`compare` refuses runs measured differently: another workload, upstream
revision, judge, answering model or tokenizer, or different questions. It
reports the headline and per-type deltas, the questions that started and
stopped passing, evidence recall, recall latency, and median tokens over the
questions both runs passed (returning less is only cheaper if it still
answers, #309). It lists what differs about the system: build inputs, recall
lanes and packing, commit. Without `--noise-floor` it calls nothing real.

`calibrate` derives the noise floor from repeated runs of the same workload,
and reports pass-set stability and per-question pass rates beside it: a steady
rate can hide a different passing set each run. Scores drift by about 0.04 on
BEAM and 3 points on LongMemEval across hours, so re-run the baseline in the
same session before attributing a change.

## Publishing a version point

An official run becomes a version point by committing it without benchmark
text, adding it to the table at the top and writing its report in
[`docs/benchmark-results.md`](docs/benchmark-results.md):

```bash
context-graph-eval run beam-100k
context-graph-eval publish runs/beam-100k/<label>.json baselines/beam-100k-gliner2.json
```

`publish` drops every row's questions, answers and retrieval, keeping the
scores, metrics and latencies `compare` needs, so a later run can be compared
against the published one.

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

LongMemEval and BEAM inject fixtures directly, so they never exercise the
capture layer. The gold slice drives a **real, billed Claude Code session**
with hooks live and scores what was actually recorded — the only eval coverage
for subagent nesting (#308). Its one question asks for a fact that exists only
inside a subagent. It is Tier 2: authored for this project, and never blended
with the adopted benchmarks (#303).

```bash
context-graph-eval gold-slice
context-graph-eval run gold-slice --rubric-judge anthropic:claude-sonnet-4-5-20250929
```

The planting prompt must not contain the fact, the fact must be unguessable
(a hex revision), and the prompt must contain no literal `SKILL.md` path
(#293). The run refuses unless the fact is planted, nested under an `Agent`,
and absent from top level. Hooks are pointed at the gold slice's instance
through a throwaway config named by `CONTEXT_GRAPH_CONFIG` (ADR 0003). No
upstream judge has a template for it, so the rubric judge decides it.

## Known limitations

- **Building dominates cost.** GLiNER2 on the CPU is most of a build: about
  8 hours per 1,000 LongMemEval sessions, or 8 hours for BEAM 100K, on a
  6-core machine.
- **The noise floor is large relative to deltas**; calibrate on the workload
  before trusting a small change.
- **Tier 2 has one question.**

## Development

```bash
./scripts/dev-memgraph.sh test context-graph-eval
```

Tests read `EVAL_MEMGRAPH_URL` (default `bolt://localhost:7689`) and skip the
end-to-end ones when no instance is reachable.
