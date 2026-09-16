# Context Graph Eval

Context Graph Eval measures whether Memory-Tier output can answer recall
questions. It is separate from the production pipeline it measures.

## Language

**Corpus**:
The set of questions with known-correct answers that a run is scored against.
Stored as deepeval `Golden` records in committed JSONL, never in Memgraph. Eval
mutates its staged graph, so keeping the answer key there could invalidate it.
_Avoid_: Dataset, test set, benchmark (a benchmark is an upstream *source* for a
corpus, not the corpus itself)

**Tier**:
The kind of recall being tested. **Tier 1 (Mechanical Recall)** comes from an
upstream benchmark. **Tier 2 (Organizational Recall)** is authored for this
project. Scores remain separate.
_Avoid_: Level, phase, stage

**Gold Slice**:
Tier 2 cases staged through a real agent session instead of direct fixture
injection. They cover capture hooks, Skill usage, and subagent nesting. Cases are
grouped by where each fact lives in the graph.
_Avoid_: Smoke test, integration test, golden test

**Coverage**:
Whether an LLM judge says the graph answer satisfies the answer key. Coverage is
a hard gate: failed questions are excluded from Efficiency ranking.

An ordinary answer must contain the required facts and use non-empty retrieval.
An **Abstention** answer must decline; empty retrieval is then correct.
_Avoid_: Accuracy, recall (recall is the thing being measured, not this metric),
score

**Abstention**:
A question whose correct answer is that the information is *not* in the graph.
It measures not fabricating, so a confident specific answer is the failure and
an empty retrieval is correct behaviour rather than a miss.

Reported separately because ordinary Coverage rules invert. Empty retrieval is
correct, and a refusal should not need to repeat contrastive details from the
answer key.
_Avoid_: Negative, trap, null question, unanswerable (the question is perfectly
answerable — "it is not in memory" is the answer)

**Efficiency**:
Token count of the retrieved payload. It does not count the agent's own usage and
only ranks questions that passed Coverage.
_Avoid_: Cost, latency, performance

**Retrieval**:
The step under test: answering a question from the graph. In v1, an agent receives
the schema and writes read-only Cypher. There is no ranking, query template, or
vector search yet.

Only Retrieval is read-only. It cannot alter the graph or access distillation
storage, including cached LLM responses that may contain answers.
_Avoid_: Query, search, lookup (each names a mechanism this deliberately has not
chosen yet)

**Eval Batch**:
One run over a corpus against a dedicated Memgraph instance. The unit of
isolation — not the individual question, because a batch-wide graph is what
gives retrieval distractors to get wrong.

The instance is wiped and re-staged by default. A batch may reuse an existing
distilled graph to reduce cost. Reuse fails early if that graph does not contain
the sessions needed by the corpus.
_Avoid_: Test run, suite, epoch

**Noise Floor**:
The smallest score change that repeat-run calibration can distinguish from
noise. Without one, comparisons report "cannot tell."
_Avoid_: Threshold, tolerance, margin of error

**Pass Stability**:
Whether the same questions pass across repeated runs, not only the same number.
A steady rate can hide a different passing set each time. Report Pass Stability
beside Noise Floor. When stability is low, Coverage deltas may be resampling and
Efficiency medians compare different question sets.
_Avoid_: Variance, flakiness (flakiness suggests a defect to fix; this is a
property to measure and report)

## Relationships

- An **Eval Batch** writes fixtures to a dedicated Memgraph instance, runs Session
  Reconciliation, retrieves answers, then scores them. Retrieval and scoring do
  not modify the staged graph. A reused graph skips fixture and reconciliation
  stages.
- **Retrieval** reads through a read-only surface. It must never be able to
  alter the graph it is scored against.
- The **Corpus** lives outside Memgraph, deliberately: the instrument is kept
  out of the thing it measures.
- Eval-triggered Session Reconciliation uses a larger local embedding model and
  a higher LLM merge-summary threshold to control batch cost. It uses the same
  mechanism and graph shape as production, but different settings (map #297).

## Flagged ambiguities

- "benchmark" is ambiguous between an upstream dataset (LongMemEval) and this
  package's own scoring. Resolved: upstream is a *source*; what this package
  runs is an **Eval Batch** against a **Corpus**.
- "score" flattens two different things. Resolved: **Coverage** is judged and
  gates; **Efficiency** is counted and ranks within the gate. There is
  deliberately no blended number.
- "memory" is Sessions Graph's term. Eval invokes reconciliation while staging a
  batch but does not define memory semantics.
- Whether eval cost tuning changes what is measured. Resolved: it keeps the same
  code path and graph shape, but results describe the tuned config, not exact
  production defaults.
