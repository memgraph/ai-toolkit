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

hygm is becoming the general component for graph models shared across
projects (sql2graph and the rest of the Memgraph ecosystem), with
language-neutral file formats for the model and its mappings; see
[docs/contract.md](docs/contract.md) (draft, tracked in #693).

## The model file

A model file is YAML or JSON in one of two format versions; both are read.
Version 1 is every model written before version 2 existed (`entity_types`,
node-level `identity`). Version 2 adds properties, a type hierarchy, keys and
IRIs:

```yaml
version: 2
node_types:
  - label: Party
    description: any person or organization the shop deals with
  - label: Customer
    description: a party that has placed an order
    parents: [Party]                 # a Customer node is also a Party
    key: [customer_id]               # identifies one customer across sources
    properties:
      - {name: customer_id, type: string, required: true}
      - {name: tier, values: [standard, gold]}
    profiles: {extraction: {identity: global}}
relation_types:
  - {label: placed, description: placed the order, start_labels: [Customer], end_labels: [Order]}
```

`model_to_mapping` writes version 1 whenever a model fits in it, so models
stored before version 2 serialize exactly as they did. Unknown keys are
refused at every level. Every refusal is a `ModelError` (a `ValueError`) whose
`codes` come from the contract's error table, and
[`conformance/`](conformance/) holds the fixtures any reader, in any language,
must agree with. The JSON Schemas ship in the package (`hygm.model_schema(2)`).

Readers that can't honor a feature yet refuse models that use it:
unstructured2graph refuses `parents`, derivation refuses `parents`,
`properties` and `key`, and sessions-graph doesn't store a model that needs
version 2 until older clients can be turned away (#698).

`agents/sql2graph` has its own HyGM today. It will move onto this package and
drop its own (#701, #702). The validation types here are lifted from it in the
same shape, so that move is a type swap.
