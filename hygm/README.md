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
  `rdfs:subClassOf` a named class its `parents`, `owl:ObjectProperty` a
  relation type, and `rdfs:domain`/`rdfs:range` (one class or an
  `owl:unionOf`) the start/end labels. `owl:DatatypeProperty` becomes a
  property on its domain class(es), typed from its XSD range (unlisted types
  are `string`; an `owl:oneOf` range gives `values`); `owl:hasKey` the `key`,
  its properties required; a functional property with an at-least-one
  cardinality restriction is required; every term keeps its `iri`. Other
  axioms are ignored. Needs `hygm[owl]` (rdflib).
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

## Evolution

A model changes by operations: pure functions from a model to a model, each
with a JSON form (`{"op": "rename_node_type", "label": "Org", "to": "Company"}`,
read and written by `operation_from_mapping` and `operation_to_mapping`).

```python
from hygm import AddProperty, Property, RenameNodeType, SetKey, apply, diff

changed = apply(
    model,
    [
        RenameNodeType("Org", "Company"),
        AddProperty("node", "Company", Property("company_id", required=True)),
        SetKey("Company", ("company_id",)),
    ],
)
assert apply(model, diff(model, changed)) == changed
```

- **Types:** `AddNodeType`, `RemoveNodeType`, `RenameNodeType`, `MergeNodeTypes`
  and the same four for relation types.
- **Properties** (on a node or relation type): `AddProperty`, `RemoveProperty`,
  `RenameProperty`, `SetProperty`.
- **Settings:** `SetDescription`, `SetIri`, `SetParents`, `SetKey`,
  `SetEndpoints`, `SetProfile` (one profile's settings at the top level, on a
  type or on a property; a node type's `extraction` includes `identity`).
- **Order:** `OrderNodeTypes`, `OrderRelationTypes`, `OrderProperties`.

`apply` validates only the result, so a list may pass through invalid models:
removing a type leaves what names it for a later operation to change. Renames
and merges carry through to parents, endpoints, keys and label-valued profile
settings. An operation that doesn't apply raises `ModelError` with
`unknown_type`, `undeclared_property`, `duplicate_label`, `duplicate_property`
or `invalid_operation`; reading an unknown `op` raises `unknown_operation`.

`diff(a, b)` can't tell a rename from a removal and an addition, so it emits
those; a caller that knows better records its own operations. Derivation does:
`Derivation.operations` turns the current model into the candidate, and each
changelog `Change` (the record sessions-graph stores) says what it did through
`Change.operations()`. A retire is a removal (its definition goes to the pool,
derivation's store), a restore an addition, and a veto names a merge that
wasn't applied. Moving the data a change implies is each consumer's job.
## Mappings

A mapping file binds a model's types to the sources their data comes from.
It is a separate file, so one model can have many mappings, and a model with
no mapping is complete. Sources are names with a `kind` (`table` or `graph`),
never connection details:

```yaml
version: 1
model: shop.model.yaml               # relative to this file
sources:
  - {name: crm, kind: table}
  - {name: graph, kind: graph}
nodes:
  - type: Customer
    source: crm
    table: {schema: public, name: customers}
    id: [customer_id]                # columns identifying a row
    properties:
      - {name: customer_id, column: customer_id}
      - {name: loyalty_points, column: attributes, path: $.points}
  - type: Customer                   # bound twice, so Customer needs a key
    source: graph
    graph: {label: Client}           # the source's own label, to read
relations:
  - type: placed
    source: crm
    table: {name: orders}
    from: {type: Customer, columns: [customer_id]}
    to: {type: Order, columns: [order_id]}
  # or, with no rows: a join on equal endpoint properties
  # - {type: placed, join: {from: {type: Customer, property: customer_id}, to: {type: Order, property: customer_ref}}}
```

```python
from hygm import load_mapping, mapping_to_mapping

model, mapping = load_mapping("shop.mapping.yaml")
```

`load_mapping` reads the model with `ManualStrategy` and checks the mapping
against it (`check_mapping`): declared types, sources and properties; the
binding each source kind allows; a `key` for a type bound in several sources;
endpoints the relation type admits; and `from`/`to` columns that match the
endpoint's `id` in the same source (or its `key` when it isn't bound there).
A mapping never declares labels: a bound node carries its type and its
ancestors' labels. A property binds to one column, optionally through a JSON
`path`, and nothing else. Refusals are `ModelError`s with the contract's
codes; the schema ships as `hygm.mapping_schema()`.

Other tools keep their own formats. A `Renderer` (`render(model, mapping)`)
turns a model and mapping into one of them, losslessly, and refuses whatever
its target can't express rather than dropping it.
## The Memgraph plan

`memgraph_plan(model)` derives the constraints, indexes and enums a Memgraph
needs to hold a model, in declaration order; `apply_plan(plan, run)` creates
the ones that don't exist yet, so applying it twice is a no-op. `run` is your
client: a callable that runs one Cypher string and returns its rows, so hygm
takes no Memgraph dependency.

```python
from hygm import apply_plan, enum_for, labels_for, memgraph_plan

apply_plan(memgraph_plan(model), lambda q: [r.data() for r in session.run(q)])
labels_for(model, "VipCustomer")  # ("VipCustomer", "Customer", "Party"): set them all on write
enum_for(model, "Customer", "tier")  # "Customer_tier": write ToEnum("Customer_tier", "gold")
```

| Model | Plan |
|---|---|
| a type's `key` | uniqueness constraint (composite for several properties) and a label-property index |
| `required: true` | existence constraint |
| `type` | `IS TYPED` constraint: `string` STRING, `integer` INTEGER, `float` and `decimal` FLOAT, `boolean` BOOLEAN, `date` DATE, `json` MAP; `datetime` none (Local or Zoned) |
| `values` | an enum `<Label>_<property>`, and the property `IS TYPED ENUM` |
| `parents` | no DDL; writers set `labels_for` |

Each fact is constrained once, on the label of the type that declares it: a
key on `Customer` covers `VipCustomer` nodes, which carry the `Customer`
label. Memgraph can type a property only as "some enum", not one named enum,
and has no constraints on relationship properties, so relation types add
nothing. A property already typed differently, or an enum with other values,
raises `PlanConflictError` before anything runs.

`agents/sql2graph` has its own HyGM today. It will move onto this package and
drop its own (#701, #702). The validation types here are lifted from it in the
same shape, so that move is a type swap.
