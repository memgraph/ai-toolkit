# Context Graph Eval

Measures whether recall over History and Memory can answer recall questions. Separate from the production pipeline it measures.

## Language

**Benchmark**:
Upstream source of questions and haystacks: LongMemEval-S, BEAM 100K. Adapted to the pipeline by a benchmark adapter that brings its questions, its haystack, its own judge and its headline -- nothing else differs between benchmarks.
_Avoid_: Dataset, test set

**Workload**:
Named, fixed set of a benchmark's questions and the haystack they're asked of. **Official**: the whole benchmark, uncapped; its result is what we communicate. **Dev**: a small, fixed slice for iterating; compared only with other runs of itself, never quoted.
_Avoid_: Batch, suite, split, subset

**Corpus**:
Committed JSONL of a workload's LongMemEval questions (deepeval `Golden` records), never in Memgraph. The eval builds the graph -- an answer key living there could get invalidated, and git proves two compared runs asked the same questions.
_Avoid_: Dataset, test set

**Tier**:
Kind of recall tested. **Tier 1 (Mechanical Recall)**: from an upstream benchmark. **Tier 2 (Organizational Recall)**: authored for this project (the gold slice). Separate workloads, so never blended.
_Avoid_: Level, phase, stage

**Gold Slice**:
Tier 2 workload staged through a real agent session, not injected. Covers capture hooks, Skill usage, subagent nesting.
_Avoid_: Smoke test, integration test, golden test

**Eval Instance**:
A workload's own Memgraph container, named for it, with its graph on a durable volume. Dedicated: a rebuild wipes it.
_Avoid_: Test database, dev Memgraph

**Build**:
Inject, reconcile, (derive), embed: turning a workload's haystack into the graph recall reads. Kept on the eval instance with a **manifest** fingerprinting what built it -- the questions, the build settings, the source of every module that shapes the graph. Same fingerprint: reused, or resumed if interrupted. Different: rebuilt from empty.
_Avoid_: Staging, ingestion, index

**Shard**:
The questions one build holds when a workload's whole graph doesn't fit in memory (official LongMemEval-S: 25 of 20). Independent: each question is its own user.
_Avoid_: Batch, chunk, partition

**Run**:
One workload asked of its build: recall, answer, judge. Saved as one row per question plus what it was measured against.
_Avoid_: Eval batch, test run, epoch

**Score**:
A question's verdict from its benchmark's own judge, 0 to 1. LongMemEval: right (1) or wrong (0), LongMemEval's judge. BEAM: rubric items at 1, 0.5 or 0, BEAM's judge. **Passed**: at least 0.5. A question its judge couldn't score is **unjudged**, left out of every mean, never 0.
_Avoid_: Accuracy (that's LongMemEval's headline), coverage, recall

**Headline**:
The one number a benchmark reports from its scores: LongMemEval averages questions, BEAM averages abilities.
_Avoid_: Overall, total, score

**Version Point**:
An official run, committed to `baselines/` without benchmark text, behind a row of the README's official numbers.
_Avoid_: Release, benchmark result

**Abstention**:
Question whose correct answer = info *not* in graph. Measures not-fabricating: confident specific answer = failure, empty retrieval = correct.
_Avoid_: Negative, trap, null question, unanswerable (question IS answerable — "not in memory" is the answer)

**Efficiency**:
Token count of retrieved payload. Excludes the answering model's own usage; compared only over questions that passed.
_Avoid_: Cost, latency, performance

**Recall**:
Step under test: `sessions_graph.recall`, the product read path, scoped to the asking user. The only retrieval the eval measures; every benchmark answers from it through one shared prompt. Read-only: can't alter the graph it's scored against.
_Avoid_: Query, search, lookup, strategy

**Noise Floor**:
Smallest headline change repeat-run calibration can tell apart from noise. Without one, comparisons report "cannot tell."
_Avoid_: Threshold, tolerance, margin of error

**Pass Stability**:
Whether same questions pass across repeated runs, not just same count. Steady rate can hide a different passing set each time. Report beside Noise Floor.
_Avoid_: Variance, flakiness (flakiness implies a defect to fix; this is a property to measure/report)

## Relationships

- A **Workload** names one **Benchmark**, an **Eval Instance**, and (if it doesn't fit one graph) its **Shards**.
- A **Run** finds the instance's **Build**: reuses it when its fingerprint matches, else rebuilds. Recall and judging never modify it.
- A **Run** of an official workload, published, is a **Version Point**. Dev runs are compared only with dev runs of the same workload.
- **Corpus** lives outside Memgraph, deliberately: instrument kept out of the thing it measures.

## Flagged ambiguities

- "benchmark" ambiguous between upstream dataset and this package's own run. Resolved: upstream = **Benchmark**; this package runs a **Workload**.
- "batch" meant three things: one eval run, a 20-question slice of LongMemEval, and a fast iteration set. Resolved: **Run**, **Shard**, dev **Workload**.
- "score" flattens two things. Resolved: **Score** is judged per question; **Efficiency** is counted, and compared only within questions that passed. No blended number.
- "memory" = Sessions Graph's term. Eval invokes reconciliation while building but doesn't define memory semantics.
