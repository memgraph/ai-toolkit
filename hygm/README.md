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
- `LlmRecommendationStrategy`: derives a model from a corpus sample. Interface
  only for now.

`validate_model` is the hard gate every model passes before anything extracts
against it. `ManualStrategy` runs it on load.

Named after, but separate from, `agents/sql2graph`'s HyGM, which is untouched.
Its validation types are lifted here in the same shape, so a later
consolidation is a type swap.
