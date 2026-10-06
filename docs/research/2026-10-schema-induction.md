# Incremental LLM-driven schema induction from a growing stream

Research for #437 (map #431). Question: how do published methods learn and
evolve a knowledge-graph schema incrementally from a growing stream of
conversations or documents, with an LLM proposing types? Four sub-questions:
merging proposals into an existing schema, deciding when to re-derive,
keeping earlier extractions consistent after a change, and evaluating a
schema without human labels.

Sources were read in their primary form (papers' full text, one vendor's
official documentation). Per this repo's convention for shared research, the
methods are described by process only; systems are not named.

## Sources at a glance

| Ref | Kind | What it contributes |
|---|---|---|
| A | Extract-then-define-then-canonicalize KG construction (2024) | Definition-embedding retrieval + LLM verification to merge relations into a growing schema |
| B | LLM taxonomy generation over minibatches for conversation logs (2024) | Propose/update/review loop over minibatches; LLM judges validated against humans |
| C | Streaming slot-schema induction for dialogues (2025) | Joint "fill existing or create new" per dialogue; sliding-window pruning of unused slots |
| D | Schema-free incremental KG construction with evolution-intent assessment (2026) | Cluster-frequency threshold to admit a type; proposal pool; soft deprecation of facts |
| E | Hierarchical type induction with constrained edit actions (2026) | Merge/split/create/reassign/modify actions over embedding neighbourhoods, iterated until edits plateau |
| F | Auditable schema induction and fusion benchmark (2026) | Conservative fusion with domain/range checks; label-free-ish schema-graph similarity metrics; ~$0.12 per source |
| G | Graph-grounded ontology induction with constrained LLM mediation (2026) | LLM only names clusters / assigns domain-range; vocabulary-saturation curve as convergence signal |
| H | Web-scale autonomous schema induction via conceptualization (2025) | Multi-level abstraction phrases per instance; types alignment vs reference schemas |
| I | Domain-agnostic generative ontology induction (2026) | Node-coverage score: generate text from the ontology and check its nodes appear |
| J | Incremental zero-shot KG construction from documents (2024) | Embedding thresholds (≈0.6 entities, ≈0.56 relations) to resolve new against global |
| K | Temporal KG for agent memory (2025) | Per-episode incremental resolution; bi-temporal edge invalidation; periodic full refresh of incremental structures |
| L | A graph-RAG vendor's ontology-evolution docs | Operation classes (declare / mechanical migrate / LLM backfill); "backfill first, schema commit last" |
| M | Property-graph schema evolution theory (2019) | Evolution as graph rewriting; instance validity via schema homomorphism |

## 1. Merging new proposals into an existing schema

Every incremental method avoids asking the LLM to rewrite the whole schema
free-form. They converge on **retrieve candidates, then let the LLM make a
bounded choice**:

- **Definition-embedding retrieval + LLM multiple choice (A, J, K).** Each
  proposed type/relation gets a one-sentence natural-language *definition*
  written by the LLM ("bornOn: the subject was born on the date given by the
  object"). The definition — not the label — is embedded, the top-k (k=5 in
  A) nearest existing schema elements are retrieved, and the LLM picks one or
  rejects all. On reject the element is **added** to the schema
  ("self-canonicalization" — the schema grows from empty). Effect in A: open
  extraction produced 529/667/204 relation types on three benchmarks; the
  canonicalized schemas were 200/225/106, at 0.87–0.96 human-judged precision.
  J resolves by plain cosine thresholds (≈0.6 entities, ≈0.56 relations on a
  3072-d embedding) with near-zero false merges on its test sets. K does the
  same for instances with a cheap deterministic path (MinHash/LSH) and an LLM
  fallback.
- **Constrained edit actions over neighbourhoods (E).** Types are clustered
  by a multi-field embedding (name, description, endpoints, evidence); for
  each neighbourhood the LLM may only emit `MergeClasses`, `SplitClass`,
  `CreateClass`, `ReassignEntities`, `ModifyClass` (and for relations
  `SetCanonicalRel`, `SetRelCls`). Representations are recomputed after each
  action and the loop runs **until edits plateau or a budget is hit**. This is
  the only source with an explicit *split* operation.
- **Conservative fusion with structural vetoes (F).** New and base schemas are
  aligned with lexical + embedding + structural signals; a merge is vetoed if
  domain/range (or event role signatures) are incompatible, and ambiguous
  (polysemous) candidates are **rejected rather than forced**. Every accepted
  mapping keeps evidence pointers so it is reversible. All LLM actions are
  restricted to a pre-mined candidate space under a strict JSON contract, with
  a deterministic mining-only fallback when LLM output is malformed.
- **Frequency-gated admission (D).** Verified relation instances are
  clustered by embedding; a cluster becomes a *candidate type* only when its
  count exceeds a threshold θ and it is semantically coherent; an independent
  LLM judge then checks completeness/generalizability and writes the type with
  its domain/range. **Rejected candidates stay in a proposal pool** and are
  re-evaluated as evidence accumulates, rather than being discarded.
- **Fill-existing-first prompting (B, C).** In the streaming dialogue method
  (C) the model is conditioned on the current schema and asked to fill
  existing slots wherever possible and create a new slot only for an important
  value no slot covers. In B, each minibatch's update prompt shows the current
  taxonomy and asks the LLM to assess it against the new data and edit it
  (merge, split, rename), with a final review pass; the authors frame this as
  SGD over the taxonomy.
- **LLM only names, clustering decides identity (G).** Surface variants are
  merged by agglomerative clustering with a distance threshold (cluster count
  is data-driven); the LLM is called only for naming clusters, assigning
  domain/range, and placing in hierarchy, with closed-vocabulary prompting
  and confidence flags.

Renames are generally modelled as *merge into canonical + keep alias*, not as
an in-place rename (A, E, F); only E exposes an explicit modify action that
changes an existing canonical label.

## 2. Deciding when to re-derive

Few sources state a re-derivation trigger outright; the ones that exist:

- **Per-batch update, always (B, C, D, K).** Every minibatch/dialogue/episode
  runs the merge step. Cost is bounded because the step is incremental
  (retrieve + bounded choice), not a full re-derivation.
- **Evidence-count thresholds (D).** A new type is created only when its
  cluster passes θ instances; pooled proposals re-enter when they gain
  evidence. This makes "when" a property of each candidate, not a global
  schedule.
- **Saturation curve (G).** Vocabulary growth vs corpus size shows rapid
  initial growth then deceleration; the authors read the flattening as
  convergence. They re-run the full pipeline on the accumulated corpus and
  rely on clustering to absorb known variants — no explicit rebuild-vs-patch
  rule.
- **Sliding-window disuse pruning (C).** A slot is removed if it was filled
  fewer than τ times in the last w dialogues (w=10, τ=1). Pruning is
  continuous, separate from creation.
- **Periodic full refresh of incremental state (K).** Incremental community
  assignment drifts from what a full recomputation gives; the fix is a
  periodic full refresh, i.e. cheap incremental updates punctuated by
  occasional full recomputes — structurally the same as a logarithmic
  schedule.
- **Plateau stop (E).** Within a derivation, iterate edit rounds until the
  number of edits per round flattens.

Nothing found prescribes a logarithmic (2, 4, 8, ...) schedule specifically,
but the combination "cheap incremental absorb every time + expensive full
re-derive when the novelty/saturation signal says so" is the common shape.
A natural trigger signal for us is the **out-of-schema rate**: the fraction of
spans/edges in new sessions that the permissive observe pass finds but the
current model cannot type, analogous to G's saturation curve and D's
candidate pool growth.

## 3. Keeping earlier extractions consistent after a change

Four patterns, from cheapest to most expensive:

- **Declarative-only changes need no migration (L, M).** Adding a type or a
  relation pattern, or editing a description, does not invalidate existing
  data. M formalizes this: an evolution is safe when the old instances are
  still homomorphic to the new schema, so "add" and "widen" changes can be
  applied without touching data.
- **Mechanical migration for merges/renames (L, A, E).** Renames and merges
  are rewritten in place by query (relabel nodes, retype edges), idempotent
  so a crash can be re-run. Aliases from the merge step drive the rewrite.
- **Re-extraction from source for semantic changes (L, H, E).** When a change
  alters *what a value means* (a new attribute, a type change, a split), the
  vendor docs reject mechanical coercion ("around thirty" cannot be cast to
  an integer, but can be re-extracted as 30 from text) and re-run the LLM over
  the source chunks that mention the affected type. Ordering rule: **backfill
  data first, commit the schema change last**, so a failure leaves the old
  schema consistent with the data. Re-extraction is scoped to chunks that
  mention the affected type, and a dry run reports the call count first.
  Splits in E are applied by `ReassignEntities`, i.e. re-assignment of
  existing instances, not re-extraction.
- **Soft deprecation + versioned provenance (D, K, F).** Facts are never
  deleted on change: edges carry a status (active/deprecated) or validity
  interval (K: valid-from/invalid-at on the event timeline plus
  created/expired on the ingestion timeline), with a log of which change
  caused the deprecation. F records a run manifest per schema version (model
  id, timestamp, hyperparameters, cost) and keeps evidence links on every
  mapping so changes are diffable and reversible. Standard ontology-language
  versioning annotations (prior version, backward-compatible-with, deprecated
  class/property) express the same at the schema level.

Most sources pair a cheap migration for the common case (merge/rename) with
scoped re-extraction for the rare semantic change.

## 4. Evaluating a schema without human labels

- **Compression vs fidelity (A, D).** Count of types before/after
  canonicalization and a redundancy score (pairwise similarity among
  canonical types). D reports 15% fewer relation types and 1.6–2.8 points less
  redundancy than A at equal or better extraction F1. This is cheap and
  label-free, but only meaningful paired with a fidelity check.
- **Coverage of the data (B, G, I).** B: fraction of items assignable to the
  taxonomy (>99.5%). I: *node coverage* — generate documents from the
  ontology and check each node appears in the output (95.6–100% vs 52–98%
  for a generic template). For us the equivalent is the share of observed
  spans/edges the model types (the inverse of the out-of-schema rate).
- **LLM-as-judge, validated once against humans (B, D).** B used a strong
  LLM to rate label accuracy/relevance; its agreement with human consensus
  (Cohen's κ 0.56–0.58 on accuracy) exceeded human-human agreement
  (Fleiss' κ ≈ 0.48), but a weaker model showed position bias and needed
  randomized option order. D judges each *new* fact against its source text
  at temperature 0.1 and counts only "fully supported" (Δ-precision ≥ 0.97)
  and evidence-backed deprecations (≥ 0.98). Both spot-checked the judge
  against humans once and then ran without labels.
- **Downstream task performance (F, G).** F measures the extraction F1 that a
  schema enables (0.56 with the released schemas vs 0.68–0.69 with induced
  ones). G answers competency questions through queries over the induced
  ontology (0.77–0.85 coverage). This is the closest analogue to our
  recall-outcome gate.
- **Structural consistency (E).** Domain/range consistency of instances
  against declared signatures (82.7%), and a graph-utilization metric
  (share of the graph actually used by answers).
- **Run-to-run stability (B, I).** I reports that runs vary in naming and
  granularity and that multi-example grounding reduces but does not remove
  it. B runs 10 trials and keeps the best on a validation signal rather than
  trying to make one run stable. Nobody reports a stability number
  comparable to our 84% span co-assignment agreement; selecting among runs
  by an outcome signal is the published answer to instability.

Cost reference points: F ≈ 4 LLM calls, 35 s and $0.12 per source schema;
D/A run one bounded LLM call per merge decision; our #372 derivation was
$1.85 plus ~3.7 h single-core observe.

## Applicability to our propose / consolidate / observe / prune pipeline

| Our step | Most applicable technique | Why |
|---|---|---|
| Propose | Fill-existing-first prompting (B, C): show the current model and ask only for what it fails to cover | Turns re-derivation into aggregation by construction; proposals shrink as the model saturates |
| Consolidate | Definition-embedding retrieval + bounded LLM choice (A), with domain/range veto and "reject when ambiguous" (F) | Replaces one large merge call with many small, auditable decisions; directly addresses the #360 near-synonym problem; vetoes stop merges that change a relation's range (#359) |
| Consolidate (splits) | Constrained edit actions with plateau stop (E) | Only published way to get splits; bounded vocabulary keeps it gate-checkable |
| Observe | Unchanged (local GLiNER2), but report out-of-schema rate | Becomes the re-derive trigger and the coverage metric |
| Prune | Frequency threshold with a proposal pool (D), sliding-window disuse (C) | Pooled-not-deleted candidates fix the #372 failure where value relations were pruned on thin evidence and then lost |
| Schedule | Incremental absorb at each 2^k checkpoint; full re-derive only when out-of-schema rate or proposal-pool size crosses a threshold (D, G, K) | Fits the logarithmic schedule; the expensive path runs only when saturation says the model is stale |
| Migration | Alias-driven relabel for merges; scoped re-extraction (backfill first, commit model last) only for splits and new value types (L, M); soft-deprecate rather than delete (D, K) | Re-extraction is local GLiNER2 compute; recall can read old edges through the alias map |
| Gate / eval | `validate_model()` + downstream outcome (F, G) as the hard gate; coverage, redundancy, LLM-judged support of new facts (B, D) as reporting-only; pick the best of several seeds rather than requiring stability (B) | Matches the map's "automatic hard gate + reporting-only signals" rule |

Open gaps none of the sources close: no published method handles
value-typed attributes (our `25:50` → `Duration` loss) through an incremental
schema loop — the vendor docs treat attributes as a separate,
re-extraction-backed operation, which is the closest match. No source gives
a principled re-derive schedule; ours would be the first to fix 2^k
checkpoints and should report the saturation curve to justify it.
