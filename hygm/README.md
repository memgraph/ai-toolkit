# hygm

A general graph model, or ontology, component: node types with a type
hierarchy, properties and keys, relation types with the node labels they may
connect (`start_labels`/`end_labels`, OWL's `rdfs:domain`/`rdfs:range`), and
the files, checks and tools around them. hygm knows no consumer, domain or
application; the projects that model graphs with it (unstructured2graph,
sessions-graph, sql2graph and other tools in the Memgraph ecosystem) depend on
it, never the other way round. The contract is
[docs/contract.md](docs/contract.md) (draft, tracked in #693 and #721).

```python
from hygm import ManualStrategy, validate_model

model = ManualStrategy().create_model("ontology.yaml")
```

Strategies:

- `ManualStrategy`: a hand-authored YAML or JSON file (see its docstring for the shape).
- `OwlImportStrategy`: a standard OWL ontology. `owl:Class` becomes a node type,
  `rdfs:subClassOf` a named class its `parents`, `owl:ObjectProperty` a
  relation type, and `rdfs:domain`/`rdfs:range` (one class or an
  `owl:unionOf`) the start/end labels. `owl:DatatypeProperty` becomes a
  property on its domain class(es), typed from its XSD range (unlisted types
  are `string`; an `owl:oneOf` range gives `values`, `xsd:minInclusive` and
  `xsd:maxInclusive` the bounds); `owl:hasKey` the `key`, its properties
  required. Cardinality and `owl:hasValue` restrictions set counts and
  `value`, on an object property per type (`relation_restrictions`, with
  `owl:allValuesFrom`); disjointness, inverses, symmetric and functional
  properties, language-tagged labels and `skos:altLabel`, named individuals,
  the `owl:Ontology` header and its `owl:imports` of local files come across
  too; every term keeps its `iri`. Equivalence and other axioms are ignored.
  `model_from_owl` reads a document held in memory. Needs `hygm[owl]` (rdflib).

`model_to_owl(model, format="turtle")` writes a model as OWL, the same table
backwards; what OWL has no construct for (`abstract`, `default`, `unit`,
profiles, relation counts and properties) goes in `hygm:` annotations. Read
back, it is the same model up to order, which OWL doesn't keep. Types,
properties and individuals without an IRI get one under `base`
(`urn:hygm:model#` by default), and the import leaves those out again.

`validate_model` is the hard gate every model passes before anything uses it.
`ManualStrategy` runs it on load.

## Profiles

A profile carries what one kind of consumer needs from a model: settings on
the model, a type or a property, under the profile's name. hygm holds the
registry (`REGISTERED_PROFILES`, `OWNERS`) and the shape of each profile's
settings, and checks them when it reads a file; a private profile is named
`x_...`. What a setting *means* belongs to its owner:

| Profile | Owner |
|---|---|
| `extraction` (`identity`, `value`) | unstructured2graph, `unstructured2graph.profile` |
| `conversation` (`speaker`, `other_person`) | unstructured2graph, `unstructured2graph.profile` |
| `learning` (`protected`, `catch_all`) | sessions-graph, `sessions_graph.learning` |

A version-1 file (below) may imply profiles its owner used to hardcode. The
reader and writer take that rule as one hook, `version_1_profiles`:

```python
from hygm import ManualStrategy, model_to_mapping

model = ManualStrategy(version_1_profiles=rule).create_model("legacy.yaml")
model_to_mapping(model, version_1_profiles=rule)  # still version 1 if nothing else changed
```

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

A consumer that can't honor a feature yet refuses a model that uses it rather
than dropping it silently (contract, section 8).

A model may say what it is for in an `ontology` block (scope, competency
questions, maintainers, naming conventions), and build on other models with
`imports`, each a path relative to the importing file:

```yaml
version: 2
ontology:
  name: wine
  competency_questions: [Does Cabernet Sauvignon go well with seafood?]
imports:
  - {path: food.model.yaml}          # its types become parents and endpoints here
node_types:
  - {label: Wine, description: a drink made from grapes}
relation_types:
  - {label: served_with, description: goes with, start_labels: [Wine], end_labels: [Seafood]}
```

`ManualStrategy` resolves imports transitively; `model_to_mapping` writes the
importing file back with its `imports`, never inlining them. Given a model as
data, `model_from_mapping` needs a `resolver` for its imports.

Node types may be `abstract` (no direct instances) and `disjoint_with` other
types; `model.disjoint(a, b)` answers disjointness both ways, through
ancestors. Types, relation types and properties take `synonyms` and `names`
(display names by BCP 47 language tag). A synonym naming another type is
refused (`duplicate_name`), as are two disjoint types sharing a descendant
(`disjoint_conflict`).

Properties take facets: `min_count`/`max_count` (`required` is `min_count:
1`; a `max_count` above 1, or `null`, makes the property a list), `default`,
a fixed `value`, `minimum`/`maximum` on numbers and dates (ISO 8601 strings),
and a `unit`. A subtype redeclares an inherited property to narrow it, stating
only what it narrows; `properties_of` returns the effective facets:

```yaml
- label: WhiteWine
  parents: [Wine]
  properties:
    - {name: tannin_level, max_count: 0}       # white wines have none
- label: DessertWine
  parents: [Wine]
  properties:
    - {name: sugar, values: [sweet], value: sweet}
```

Relation types take counts per start node (`min_count`/`max_count`),
`inverse_of` and `symmetric`, and a node type narrows a relation for its own
instances with `relation_restrictions` (narrower end labels, tighter counts).
`model.inverse(label)` and `model.relation_for(label, relation)` answer them,
restrictions inherited down the hierarchy.

A model may carry `individuals`, instances checked against their types'
facets and relations (inverse and symmetric relationships are implied, not
repeated):

```yaml
individuals:
  - {id: bordeaux, type: WineRegion, properties: {name: Bordeaux}}
  - {id: chateau_margaux, type: Winery, properties: {name: Château Margaux}, relations: {located_in: [bordeaux]}}
```

`seed_statements(model)` turns them into idempotent, parameterized `MERGE`
statements (keyed by the type's key, else an `id` property), and
`apply_seed(statements, run)` runs them; the plan never seeds on its own.

The Memgraph plan types a multi-valued property `LIST`; Memgraph can't check
its elements' type, the counts or the bounds, so writers do.

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
  type or on a property).
- **Ontology keys** (contract 4.11–4.16): `SetOntology`, `SetImports`,
  `SetAbstract`, `SetDisjointWith`, `SetSynonyms`, `SetNames`,
  `SetRelationRestrictions`, `SetRelationCounts`, `SetInverse`,
  `SetSymmetric`. Property facets change through `SetProperty`.
- **Individuals:** `AddIndividual`, `RemoveIndividual`, `SetIndividual`.
- **Order:** `OrderNodeTypes`, `OrderRelationTypes`, `OrderProperties`,
  `OrderIndividuals`.

`apply` validates only the result, so a list may pass through invalid models:
removing a type leaves what names it for a later operation to change. Renames
and merges carry through to parents, endpoints, keys and label-valued profile
settings. An operation that doesn't apply raises `ModelError` with
`unknown_type`, `undeclared_property`, `duplicate_label`, `duplicate_property`,
`unknown_individual`, `duplicate_individual` or `invalid_operation`; reading an
unknown `op` raises `unknown_operation`.

`SetImports` changes only which models are imported; the types and
individuals they declare are added and removed by their own operations, and a
result missing or changing one is refused. Reading a `set_imports` back from
JSON takes a `resolver`, as reading a model file with imports does.

`diff(a, b)` can't tell a rename from a removal and an addition, so it emits
those; a caller that knows better records its own operations. Moving the data
a change implies is each consumer's job.

`agents/sql2graph` has its own HyGM today. It will move onto this package and
drop its own (#701, #702).
