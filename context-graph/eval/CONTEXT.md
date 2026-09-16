# Context Graph Eval

Measures whether Memory-Tier output can answer recall questions. Separate from the production pipeline it measures.

## Language

**Corpus**:
Set of questions with known-correct answers a run is scored against. Stored as deepeval `Golden` records in committed JSONL, never Memgraph. Eval mutates its staged graph — answer key living there could get invalidated.
_Avoid_: Dataset, test set, benchmark (benchmark = upstream *source* for a corpus, not the corpus itself)

**Tier**:
Kind of recall tested. **Tier 1 (Mechanical Recall)**: from upstream benchmark. **Tier 2 (Organizational Recall)**: authored for this project. Scores stay separate.
_Avoid_: Level, phase, stage

**Gold Slice**:
Tier 2 cases staged through a real agent session, not direct fixture injection. Covers capture hooks, Skill usage, subagent nesting. Grouped by where each fact lives in the graph.
_Avoid_: Smoke test, integration test, golden test

**Coverage**:
Whether LLM judge says graph answer satisfies answer key. Hard gate: failed questions excluded from Efficiency ranking.

Ordinary answer: must contain required facts + use non-empty retrieval. **Abstention** answer: must decline; empty retrieval then correct.
_Avoid_: Accuracy, recall (recall = the thing measured, not this metric), score

**Abstention**:
Question whose correct answer = info *not* in graph. Measures not-fabricating: confident specific answer = failure, empty retrieval = correct.

Reported separately: ordinary Coverage rules invert here. Empty retrieval correct; refusal shouldn't need to repeat contrastive details from answer key.
_Avoid_: Negative, trap, null question, unanswerable (question IS answerable — "not in memory" is the answer)

**Efficiency**:
Token count of retrieved payload. Excludes agent's own usage; only ranks questions that passed Coverage.
_Avoid_: Cost, latency, performance

**Retrieval**:
Step under test: answering a question from graph. v1: agent gets schema, writes read-only Cypher. No ranking, query template, vector search yet.

Only Retrieval is read-only. Can't alter graph or access distillation storage, incl. cached LLM responses that may hold answers.
_Avoid_: Query, search, lookup (each names a mechanism not yet chosen)

**Eval Batch**:
One run over a corpus against dedicated Memgraph instance. Unit of isolation — not the question, since batch-wide graph gives retrieval distractors to get wrong.

Instance wiped + re-staged by default. Batch may reuse existing distilled graph to cut cost. Reuse fails early if that graph lacks sessions the corpus needs.
_Avoid_: Test run, suite, epoch

**Noise Floor**:
Smallest score change repeat-run calibration can tell apart from noise. Without one, comparisons report "cannot tell."
_Avoid_: Threshold, tolerance, margin of error

**Pass Stability**:
Whether same questions pass across repeated runs, not just same count. Steady rate can hide a different passing set each time. Report beside Noise Floor. Low stability -> Coverage deltas may be resampling, Efficiency medians compare different question sets.
_Avoid_: Variance, flakiness (flakiness implies a defect to fix; this is a property to measure/report)

## Relationships

- **Eval Batch** writes fixtures to dedicated Memgraph instance, runs Session Reconciliation, retrieves answers, scores them. Retrieval + scoring never modify staged graph. Reused graph skips fixture + reconciliation stages.
- **Retrieval** reads through read-only surface. Must never alter the graph it's scored against.
- **Corpus** lives outside Memgraph, deliberately: instrument kept out of the thing it measures.
- Eval-triggered Session Reconciliation uses larger local embedding model + higher LLM merge-summary threshold to control batch cost. Same mechanism + graph shape as production, different settings (map #297).

## Flagged ambiguities

- "benchmark" ambiguous between upstream dataset (LongMemEval) and this package's own scoring. Resolved: upstream = *source*; this package runs an **Eval Batch** against a **Corpus**.
- "score" flattens two things. Resolved: **Coverage** judged + gates; **Efficiency** counted + ranks within gate. No blended number.
- "memory" = Sessions Graph's term. Eval invokes reconciliation while staging a batch but doesn't define memory semantics.
- Does eval cost tuning change what's measured? Resolved: no — same code path + graph shape, but results describe tuned config, not exact production defaults.
