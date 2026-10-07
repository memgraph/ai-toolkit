# hygm

A shared graph schema/ontology model: node types, relation types with the
node labels they may connect (`start_labels`/`end_labels`, OWL's
`rdfs:domain`/`rdfs:range`), and how each node type's mentions merge into
nodes (`identity`). Producing packages (`unstructured2graph`) extract and
validate against it.

```python
from hygm import ManualStrategy, validate_model

model = ManualStrategy().create_model("ontology.yaml")
```

Strategies:

- `ManualStrategy`: a hand-authored YAML file (see its docstring for the shape).
- `OwlImportStrategy`: a standard OWL ontology. `owl:Class` becomes a node type,
  `owl:ObjectProperty` a relation type, and `rdfs:domain`/`rdfs:range` (one
  class or an `owl:unionOf`) the start/end labels. Needs `hygm[owl]` (rdflib).
- `LlmRecommendationStrategy`: grows a model from a delta of new sessions. One
  `derive()` run takes the current model and returns a candidate
  (`Derivation`) with a changelog, observation counts and a pool of retired
  types:
  - LLM propose calls over batches of the delta, told what the model already has;
  - one consolidate call that may merge synonyms (vetoed on protected labels,
    differing identities or incompatible endpoints);
  - a permissive observe pass through a caller-supplied `Observer`;
  - an LLM prune that sets new relations' endpoints from what was observed;
  - retirement to the pool below a share threshold (once there's enough
    evidence) or past the active-type cap (~40 node, ~60 relation types).

  The core, the catch-alls and pinned labels are never merged away or retired.
  You supply the LLM as `llm(system, prompt, schema) -> reply`; hygm takes no
  SDK.

The generic default model ships with the package:

```python
from hygm import CATCH_ALL_LABELS, CORE_LABELS, default_model, with_core

model = default_model()  # core + Organization/Location/Event + catch-alls
supplied = with_core(my_model)  # a supplied model with any missing core type added
```

- **The fixed core** (`CORE_LABELS`): `User`, `Person`, and the value types
  (`VALUE_LABELS`: `Duration`, `Quantity`, `Money`, `Date`, `TimeWindow`).
  A learned model never merges or retires these.
- **The generic layer:** `Organization`, `Location` and `Event`.
- **The catch-alls** (`CATCH_ALL_LABELS`): `Topic` and `Artifact`. They collect
  whatever has no better type, so the share of mentions landing in them
  measures how well a model fits.
- **Relations:** generic user-centred relations, plus at least one relation
  into every value type, so value facts always have somewhere to land.

`validate_model` is the hard gate every model passes before anything extracts
against it. `ManualStrategy` runs it on load.

Named after, but separate from, `agents/sql2graph`'s HyGM, which is untouched.
Its validation types are lifted here in the same shape, so a later
consolidation is a type swap.
