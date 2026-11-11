# The hygm contract

> Status: Draft, nothing implemented
> Scope: hygm as the one general component for graph models (ontologies),
> their mappings onto sources, and what follows from them, shared by every
> project that models a graph, in this repo and outside it
> Consumers today: `unstructured2graph` (text extraction), `sessions-graph`
> (stored and learned ontologies). Planned: `agents/sql2graph` (which will
> depend on hygm and drop its own HyGM), and other tools in the Memgraph
> ecosystem, including ones not written in Python
> Background: the model follows common ontology design practice, as in Noy
> and McGuinness, [Ontology Development 101](https://protege.stanford.edu/publications/ontology_development/ontology101.pdf)

---

## 1. Why

Several projects need the same thing: a description of what a graph holds,
where its data comes from, and how that description changes over time.
Each has built its own:

| Project | Has today | Missing |
|---|---|---|
| `hygm` | node and relation types, endpoints, mention identity; manual, OWL and LLM-from-text producers | properties, hierarchy, mappings, physical plan |
| `agents/sql2graph` (`core/hygm`, ~6.7k lines) | properties, sources (table/column per element), indexes, constraints, enums, edit operations | a format anyone else reads, hierarchy, descriptions as a contract |
| `sessions-graph` | versioned models stored in users' graphs, learned changes (add, merge, retire, restore) | a change vocabulary shared with anyone else |
| other Memgraph ecosystem tools | their own mappings of labels onto sources | a vocabulary shared between graphs, a type hierarchy, descriptions for agents, OWL import |

So the same `Customer` gets modeled four times, in four shapes, and none of
them can read the others. hygm becomes the component they all share. That
only works if hygm is a **contract**, not a Python class: specified file
formats with defined semantics, machine-readable schemas, and shared
conformance fixtures that a non-Python reader tests against as easily as a Python
one.

## 2. Principles

1. **The files are the contract.** The Python package is one reader and
   writer. Readers in other languages must not depend on Python.
2. **Layered.** Each layer (section 3) depends only on the layers below it.
   A consumer takes the layers it needs and nothing else.
3. **A domain-free core.** The ontology layer knows nothing about
   conversations, text, SQL or any database. Whatever only one kind of
   consumer needs lives in a profile (layer 2) or a mapping (layer 3).
4. **Logical, not physical.** The model says what exists. Where it lives is
   a mapping. How a target stores it is a plan.
5. **Fail loud.** A reader refuses what it doesn't understand: an unknown
   key, an unknown version, a feature it doesn't implement. It never drops
   meaning silently.
6. **Nothing breaks.** Every model file valid today stays readable with the
   same meaning, including the ontology versions sessions-graph has stored
   in users' graphs.
7. **No secrets.** No hygm file holds a connection string, a credential or
   a host. Mappings name sources; consumers configure how to reach them.

## 3. The layers

```
 6  Evolution      versions, diffs and operations on a model
 5  Producers      manual, OWL, SQL catalog, LLM over text, LLM over a schema
 4  Plan           what a target needs: Memgraph constraints, indexes, labels
 3  Mapping        where each type and property comes from (a separate file)
 2  Profiles       what one kind of consumer needs, inside the model file
 1  Ontology       types, properties, hierarchy, keys, descriptions
```

Layers 1–3 are file formats and are normative here. Layers 4–6 are behavior
of the hygm package; this document fixes their shape and leaves their
details to their own specs.

| Consumer | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| unstructured2graph | ✓ | `extraction` | | | | |
| sessions-graph | ✓ | `extraction`, `conversation`, `learning` | | | LLM over text | ✓ |
| sql2graph | ✓ | | ✓ | Memgraph | SQL catalog, LLM over a schema | ✓ |
| Other ecosystem tools | ✓ | | ✓ or their own format (12.1) | | | |
| Write gate (#159, proposed) | ✓ | | | Memgraph | | |

## 4. Layer 1: the ontology (model file)

### 4.1 Format

A model is a mapping, written as YAML for hand-authoring or JSON for
interchange. Every reader accepts both. Keys are `snake_case`.

```yaml
version: 2

node_types:
  - label: Party
    description: Any person or organization the shop deals with
    properties:
      - {name: name, type: string, description: Display name}

  - label: Customer
    description: A party that has placed at least one order
    parents: [Party]
    key: [customer_id]
    iri: https://example.org/shop#Customer
    properties:
      - {name: customer_id, type: string, required: true, description: "Customer id, stable across systems"}
      - {name: tier, type: string, values: [standard, gold, platinum]}
      - {name: loyalty_points, type: integer}

  - label: Order
    description: A purchase a customer made
    key: [order_id]
    properties:
      - {name: order_id, type: string, required: true}
      - {name: customer_ref, type: string}
      - {name: total, type: decimal}

relation_types:
  - label: placed
    description: The customer placed the order
    start_labels: [Customer]
    end_labels: [Order]
    properties:
      - {name: at, type: datetime}
```

### 4.2 Top level

| Key | Type | Required | Meaning |
|---|---|---|---|
| `version` | integer | yes in version 2 | Format version. A reader refuses a version it doesn't know. Absent means version 1 (section 4.9). |
| `node_types` | list of node types | yes | |
| `relation_types` | list of relation types | no, default `[]` | |
| `profiles` | map of profile name → settings | no | Model-wide profile settings (section 5). |
| `ontology` | map | no | What the ontology is for: scope, competency questions, conventions (4.11). |
| `imports` | list of `{path}` | no | Models this one builds on (4.12). |
| `individuals` | list of individuals | no | Instances the ontology carries (4.17). |

### 4.3 Node type

| Key | Type | Required | Meaning |
|---|---|---|---|
| `label` | identifier | yes | The node label. |
| `description` | string | yes, may be empty | What the type covers. Extractors steer on it; query tools and MCP servers show it to agents. |
| `parents` | list of labels | no, default `[]` | Supertypes (4.7). |
| `properties` | list of properties | no, default `[]` | Declared properties (4.8). |
| `key` | list of property names | no | Properties whose values identify one entity across sources (4.8). |
| `iri` | string | no | The type's IRI when it came from, or must round-trip to, RDF/OWL. |
| `profiles` | map | no | Per-type profile settings (section 5). |
| `abstract`, `disjoint_with`, `synonyms`, `names` | | no | 4.13. |
| `relation_restrictions` | list | no | Narrower relations for this type's instances (4.16). |

### 4.4 Relation type

| Key | Type | Required | Meaning |
|---|---|---|---|
| `label` | identifier | yes | The relationship type. |
| `description` | string | yes, may be empty | As for node types. GLiNER2 drops it (#360); other consumers don't. |
| `start_labels` | list of labels | no, default `[]` = any | `rdfs:domain`. |
| `end_labels` | list of labels | no, default `[]` = any | `rdfs:range`. |
| `properties` | list of properties | no, default `[]` | Declared relationship properties. |
| `iri` | string | no | As for node types. |
| `profiles` | map | no | Per-type profile settings. |
| `min_count`, `max_count`, `inverse_of`, `symmetric` | | no | 4.16. |
| `synonyms`, `names` | | no | As for node types (4.13). |

### 4.5 Property

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | identifier | yes | The property key. |
| `type` | property type | no, default `string` | 4.6. |
| `description` | string | no, default `""` | What the property holds. |
| `required` | boolean | no, default `false` | Every instance has a non-null value. |
| `values` | list of strings | no | The allowed values; only on a `string` property. |
| `iri` | string | no | As for node types. |
| `profiles` | map | no | Per-property profile settings. |
| `min_count`, `max_count`, `default`, `value`, `minimum`, `maximum`, `unit` | | no | Facets (4.14). |
| `synonyms`, `names` | | no | As for node types (4.13). |

### 4.6 Property types

Logical types only. Each reader maps them to its own physical types.

| Type | Memgraph | XSD (OWL import) |
|---|---|---|
| `string` | String (an enum when `values` is set, 7.1) | `xsd:string`, `xsd:anyURI`, `rdf:langString` |
| `integer` | Integer | `xsd:integer`, `xsd:int`, `xsd:long`, `xsd:short`, `xsd:byte`, `xsd:nonNegativeInteger`, … |
| `float` | Float | `xsd:float`, `xsd:double` |
| `decimal` | Float (Memgraph has no decimal) | `xsd:decimal` |
| `boolean` | Boolean | `xsd:boolean` |
| `date` | Date | `xsd:date` |
| `datetime` | LocalDateTime / ZonedDateTime | `xsd:dateTime`, `xsd:dateTimeStamp` |
| `json` | Map | `rdf:JSON` |

An XSD type not listed maps to `string`.

### 4.7 Inheritance (`parents`)

- `parents` names declared node types. The parent graph is a DAG: multiple
  parents are allowed, cycles are not.
- **Meaning:** every instance of a type is an instance of each of its
  ancestors.
- **In a property graph**, a node of type `T` carries the label `T` and the
  label of every ancestor of `T`. A `Customer` node is `(:Customer:Party)`, and
  `MATCH (p:Party)` finds it.
- **Endpoints:** a relation whose `start_labels` or `end_labels` names `T`
  admits every descendant of `T`.
- **Properties are inherited.** A descendant may redeclare an inherited
  property to narrow its facets (4.15), never to change its type. A
  property inherited through two parents takes the narrowest facets both
  allow.
- Relation types have no inheritance (`rdfs:subPropertyOf` is out of scope).

### 4.8 Properties and keys

- Property names are unique within a type, counting inherited ones.
- Declared properties are a floor, not a ceiling: an instance may carry
  undeclared properties. A consumer may be stricter. A mapping (layer 3),
  for example, may only bind declared properties.
- `required: true` is a statement about the data. Consumers that can check
  it may; the Memgraph plan turns it into an existence constraint.
- **`key`** lists the properties whose values identify one entity, so the
  same entity in two sources, or written twice, is one node. It is a
  mapping's merge rule and the plan's uniqueness constraint.
  Every key property is declared or inherited and `required: true`. A type
  has at most one key, counting its own and every ancestor's, and
  descendants inherit it.
- `key` is not mention identity. When two *mentions in text* are one node
  is the `extraction` profile's `identity` (5.3).

### 4.9 Version 1, the format before this document

Every model written before this document, including the stored ontology
versions in users' graphs, is version 1: no `version` key, `entity_types`
instead of `node_types`, and a node-level `identity`. Readers keep reading
it:

| Version 1 | Read as version 2 |
|---|---|
| `entity_types` | `node_types` |
| node `identity: X` | node `profiles: {extraction: {identity: X}}` |
| a node type labelled `Duration`, `Quantity`, `Money`, `Date` or `TimeWindow` | also `profiles: {extraction: {value: true}}` |
| a node type labelled `User`, `Person` or one of the value labels above | also `profiles: {learning: {protected: true}}` |
| a node type labelled `Topic` or `Artifact` | also `profiles: {learning: {catch_all: true}}` |
| a model declaring `User` | also top-level `profiles: {conversation: {speaker: User, other_person: Person}}` |

The implied profiles are the rules that applied to every version-1 model
before profiles existed, when they were hardcoded, so reading a version-1
file means what it always meant. They are context-graph's rules, not
hygm's (5.6): hygm itself maps a version-1 `identity` into the `extraction`
profile and implies nothing else, and the owners of the other rows apply
them when they read version-1 files. A writer that should keep such a model
in version 1 passes its implied profiles to the writer.

Version 1 has none of the other keys. A writer writes the lowest version
that can express the model: a model that only uses version 1 features, and
whose profiles are exactly the ones version 1 implies, is written exactly as
it is today. sessions-graph keeps writing what it writes
now until it stores a model that needs version 2, and must raise its graph
epoch before it does (section 11).

### 4.10 Identifiers

Labels, relation types and property names are valid Cypher identifiers
(`^[A-Za-z_][A-Za-z0-9_]*$`, as `require_valid_identifier` checks). That is
what lets every consumer interpolate them into a query safely. Comparisons
are exact and case-sensitive.

### 4.11 The ontology itself: scope and competency questions

An ontology starts from what it is for: its domain, what it will be used
for, the questions it must answer and who maintains it. The optional
top-level `ontology` block records that, so the model carries its own scope
and a consumer can test it against its purpose.

```yaml
ontology:
  name: wine
  description: Wines and the foods they pair with, for menu suggestions
  iri: https://example.org/wine
  version_info: "2026.11"
  scope:
    includes: [wine characteristics, food types, good and bad pairings]
    excludes: [cellar inventory, restaurant staff]
  competency_questions:
    - Which wine characteristics should I consider when choosing a wine?
    - Does Cabernet Sauvignon go well with seafood?
  maintainers: [the sommelier team]
  conventions: {node_case: upper_camel, property_case: snake, relation_case: snake, number: singular}
```

| Key | Type | Meaning |
|---|---|---|
| `name` | string | A short name. |
| `description` | string | The domain the ontology covers. |
| `iri` | string | The ontology's IRI (`owl:Ontology`). |
| `version_info` | string | Free-form release name (`owl:versionInfo`); not the format `version`. |
| `scope.includes`, `scope.excludes` | lists of strings | What is deliberately in and out. |
| `competency_questions` | list of strings | The questions the ontology must answer; the litmus test of its scope. |
| `maintainers` | list of strings | Who maintains it. |
| `conventions` | map | The naming conventions the lint checks (9.1). Keys: `node_case`, `property_case`, `relation_case` (each `upper_camel`, `lower_camel`, `snake` or `upper_snake`) and `number` (`singular` or `plural`). |

Every key is optional. Nothing in the block changes what the model means.

### 4.12 Imports: reusing other models

A model may build on others:

```yaml
imports:
  - {path: food.model.yaml}
```

- `path` is relative to the importing file. An imported model's node types,
  relation types and individuals become part of the importing model, as if
  declared there, and the importing model may use them as `parents`,
  endpoints and individuals' types.
- Imports are transitive and may not form a cycle (`import_cycle`).
- A label declared both in the importing model and in an import, or in two
  imports, must be the same declaration (`duplicate_label` otherwise). A
  diamond of imports is fine.
- An importing model doesn't change imported types. It refines them by
  subtyping them.
- The OWL counterpart is `owl:imports`. A reader that can't read files (one
  given a model as data) refuses a model with imports (`unknown_model`)
  unless the caller resolves them first.

### 4.13 Node types: abstract, disjoint, names

New node-type keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `abstract` | boolean | `false` | The type has no direct instances; every instance belongs to a descendant. An individual (4.17) may not have an abstract type. |
| `disjoint_with` | list of labels | `[]` | No instance is of both this type and one listed. Symmetric: declaring it on one side is enough. |
| `synonyms` | list of strings | `[]` | Other names for the same concept (synonyms are not separate classes). |
| `names` | map of language tag → string | `{}` | Display names and translations (`{en: Red wine, fr: Vin rouge}`); the `label` stays the identifier. |

Rules:

- Two disjoint types can't share a descendant, and a type can't be
  disjoint with one of its own ancestors (`disjoint_conflict`).
- A synonym may not equal another type's label or synonym, compared
  case-insensitively (`duplicate_name`). One concept, one type.

Relation types and properties take `synonyms` and `names` too.

### 4.14 Property facets

New property keys, all optional:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `min_count` | integer ≥ 0 | `0`, or `1` when `required` | The fewest values an instance has. |
| `max_count` | integer ≥ 0 or `null` | `1` | The most values an instance has. `null` means no limit; more than 1 makes the property multi-valued (a list). `0` means the type may not have the property at all, which only makes sense refining an inherited one (4.15). |
| `default` | a value of the property's type | none | Filled in when an instance is created without one. A convenience, not a constraint. |
| `value` | a value of the property's type | none | Every instance has exactly this value; for refining an inherited property on a subtype. |
| `minimum`, `maximum` | a value of the property's type | none | Inclusive bounds; only on `integer`, `float`, `decimal`, `date` and `datetime`. |
| `unit` | string | none | The unit of a numeric value (`EUR`, `ml`). Documentation; a consumer may use it. |

- `required: true` is shorthand for `min_count: 1`; a file may use either,
  and they must agree (`count_conflict`). `min_count` ≤ `max_count`.
- A multi-valued property holds a list of values of its type. Its `default`
  and `value` are lists, and `values` and the bounds apply to each element.
- `default`, `value`, `minimum` and `maximum` must be valid values of the
  type, inside `values` and the bounds (`invalid_value`). JSON has no date,
  so `date` and `datetime` values are ISO 8601 strings.
- A key property must be single-valued (`max_count: 1`).

### 4.15 Refining inherited facets in a subtype

A subtype differs from its parent by having more properties, different
restrictions on inherited ones, or different relationships. The
second needs a way to restrict an inherited property without changing its
meaning. A subtype redeclares the property, and the redeclaration may only
**narrow** it:

| Facet | A redeclaration may |
|---|---|
| `type` | not change |
| `values` | name a subset of the inherited values |
| `min_count` / `required` | raise it |
| `max_count` | lower it, keeping a multi-valued property multi-valued (a subtype's instances carry the parent's label, so a parent's list stays a list), or to `0` to forbid the property |
| `minimum` / `maximum` | raise / lower it |
| `value` | set it, to a value the inherited facets allow |
| `default` | set or change it |
| `description`, `synonyms`, `names`, `unit` | change |

Anything else is `property_conflict`. Doubly inherited properties merge by
the same rule: the result must be the narrowest facets both parents allow,
and if no value satisfies both, it is `property_conflict`.

This replaces 4.7's "only with the same `type` and `values`". The old rule
is a special case.

```yaml
- label: DessertWine
  parents: [Wine]
  properties:
    - {name: sugar, values: [dry, off_dry, sweet], value: sweet}   # every dessert wine is sweet
- label: WhiteWine
  parents: [Wine]
  properties:
    - {name: tannin_level, max_count: 0}                            # white wines don't have one
```

### 4.16 Relation facets

New relation-type keys:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `min_count`, `max_count` | as for properties | `0`, `null` | How many relationships of this type a start node has. `max_count: 1` is a single-valued slot such as a wine's `maker`. |
| `inverse_of` | relation label | none | Every `(a)-[:R]->(b)` implies `(b)-[:S]->(a)`. Declaring it on one side is enough; the two must have swapped endpoints (`inverse_conflict`). |
| `symmetric` | boolean | `false` | `(a)-[:R]->(b)` implies `(b)-[:R]->(a)`: the relation is undirected. Its start and end labels must be equal. |

A node type narrows a relation for its own instances with
`relation_restrictions`, the relation counterpart of 4.15:

```yaml
- label: FrenchWine
  parents: [Wine]
  relation_restrictions:
    - {relation: maker, end_labels: [FrenchWinery]}   # a narrower range for this type
    - {relation: grape, min_count: 1}
```

A restriction names a relation the type may start, and may only narrow its
end labels (to descendants of the inherited ones; never to none) and its
counts, which are integers. Anything else is `relation_conflict`.

### 4.17 Individuals

An ontology plus individual instances is a knowledge base. A model may carry
individuals: reference data the ontology needs, such as the wine regions, or
examples.

```yaml
individuals:
  - id: bordeaux
    type: WineRegion
    properties: {name: Bordeaux}
  - id: chateau_margaux
    type: Winery
    properties: {name: Château Margaux}
    relations: {located_in: [bordeaux]}
```

- `id` is an identifier, unique among individuals (`duplicate_individual`),
  and stands for the individual in `relations`. With an `iri` it maps to an
  `owl:NamedIndividual`.
- `type` names a declared, non-abstract node type (`unknown_type`,
  `abstract_instance`). The individual is an instance of it and its
  ancestors.
- `properties` must be declared on the type, directly or inherited
  (`undeclared_property`), and their values must satisfy every facet,
  including counts, `values`, bounds and `value` (`invalid_value`). A
  required property must be present, or filled by the type's `value` or
  `default` (`missing_value`), and a property whose `max_count` is more than
  1 takes a list. Counts include the relationships an inverse or symmetric
  relation implies.
- `relations` maps a relation type to target individual ids
  (`unknown_individual`). The relation must admit both ends and its counts
  must hold (`endpoint_not_admitted`, `count_conflict`). Inverse and
  symmetric relations are implied, not repeated.
- Individuals are data, not schema. A consumer may load them (the Memgraph
  plan seeds them on request, 7.1), show them, or ignore them.

## 5. Layer 2: profiles

A profile carries what one kind of consumer needs from the model, without
putting it into the core.

### 5.1 The mechanism

- A `profiles` map may appear at the top level and on any node type,
  relation type or property. Its keys are profile names. Each value is that
  profile's settings at that level.
- **A reader ignores the profiles it doesn't consume** and validates the ones
  it does strictly: an unknown key inside a consumed profile is an error.
  A profile never changes what the core means; a consumer that ignores it
  reads the same graph model.
- **Profile names are registered here** (5.2), each with an owner and a JSON
  Schema. A name that is neither registered nor starts with `x_` is an
  error, so `extration:` fails instead of vanishing. `x_` names are for
  experiments and private consumers; every reader ignores them unless it
  consumes them.
- A profile's own rules (5.3–5.5) are checked by its consumers, beside the
  core validation, not by it.

### 5.2 Registry

| Profile | Owner | Consumed by | What it says |
|---|---|---|---|
| `extraction` | unstructured2graph | unstructured2graph, sessions-graph | how text mentions become nodes |
| `conversation` | sessions-graph | sessions-graph, unstructured2graph | who the speaker is in a conversation |
| `learning` | sessions-graph | `LlmRecommendationStrategy` | what derivation may change |

Adding a profile is an additive change to this document.

### 5.3 `extraction`

Per node type:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `identity` | `global` \| `chunk` \| `span` | `chunk` | When two mentions are one node: one per normalized text everywhere, per chunk, or per mention. Its rules are being settled in #442; this contract carries the value and doesn't change them. |
| `value` | boolean | `false` | The type is a value (a duration, an amount), extracted as a claim about something else rather than as an entity. Today these are the fixed `VALUE_LABELS`. |

### 5.4 `conversation`

Top level:

| Key | Type | Meaning |
|---|---|---|
| `speaker` | label | The type of the person the conversation is with (`User`). |
| `other_person` | label | The type a `speaker` mention is re-typed to when it is someone else (`Person`, #358). |

Rule: every relation endpoint that admits `speaker` also admits
`other_person`. Today this is hardcoded in `validate_model` for `User` and
`Person`; it moves here, and the core gate stops applying it.

### 5.5 `learning`

Per node type:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `protected` | boolean | `false` | Derivation never merges or retires the type, and it sits outside the active-type cap. Today these are the fixed `CORE_LABELS`. |
| `catch_all` | boolean | `false` | The adoption gate measures the share of mentions landing here. Today these are the fixed `CATCH_ALL_LABELS`. |

With 5.3–5.5, the default model declares in its own file what is now
hardcoded in `labels.py`. A model for another domain declares its own value
types and catch-alls instead of inheriting the conversational ones. The
constants stay as the default model's values for code that reads them.

### 5.6 Where profile semantics live

hygm is general by itself: it knows no consumer, no domain and no
application. Of profiles it owns only the mechanism and the contract data:
the `profiles` map, the name rule (registered or `x_`), and the registry
itself, which gives each name its owner and the shape of its settings at
each level (a JSON Schema fragment shipped with hygm). So any reader checks
every registered profile's settings without its owner installed, and the
shared fixtures mean the same everywhere. What a setting *means*, and the
rules a profile adds, belong to its owner:

| Profile | Owner, where its meaning lives |
|---|---|
| `extraction` | `unstructured2graph` (`unstructured2graph.profile`): mention identity, value types |
| `conversation` | `unstructured2graph` (`unstructured2graph.profile`): the speaker and the other-person rule its mention resolver relies on |
| `learning` | `sessions-graph` (`sessions_graph.learning`): what derivation may change |

- hygm's core types carry no consumer field. Mention identity is the
  `extraction` profile's setting, read through unstructured2graph's
  accessor, not a `NodeType` attribute.
- The profiles a version-1 file implies (4.9) are the owners' rule too. The
  reader and the writer take the same hook,
  `version_1_profiles(model) -> (model profiles, profiles per node label)`:
  the reader adds what it returns to a version-1 file, and the writer keeps a
  model in version 1 when its profiles are exactly that. Without the hook,
  hygm maps a version-1 `identity` into `extraction` and implies nothing
  else.
- The conversational default model (`User`, `Person`, the value types, the
  catch-alls) and the derivation that learns a model from sessions are
  context-graph's, and live in sessions-graph.
- Nothing in hygm mentions conversations, sessions, users or a particular
  extractor.

## 6. Layer 3: mappings

A mapping binds model types to the sources their data comes from. It is a
separate file, so one model can have many mappings (a warehouse, a
transactional database, a test fixture), and a model without a mapping is
complete.

### 6.1 Format

```yaml
version: 1
model: shop.model.yaml          # path relative to this file

sources:
  - {name: crm, kind: table}
  - {name: orders_db, kind: table}
  - {name: graph, kind: graph}

nodes:
  - type: Customer
    source: crm
    table: {schema: public, name: customers}
    id: [customer_id]           # columns that identify a row
    properties:
      - {name: customer_id, column: customer_id}
      - {name: tier, column: tier}
      - {name: loyalty_points, column: attributes, path: $.points}

  - type: Customer
    source: graph
    graph: {label: Client}      # the source's own label, when it differs

  - type: Order
    source: orders_db
    table: {name: orders}
    id: [order_id]
    properties:
      - {name: order_id, column: order_id}
      - {name: total, column: total_eur}

relations:
  - type: placed
    source: orders_db
    table: {name: order_events}
    from: {type: Customer, columns: [customer_id]}
    to: {type: Order, columns: [order_id]}
    properties:
      - {name: at, column: placed_at}

  # or, with no event table, from a customer column on the order:
  # - type: placed
  #   join: {from: {type: Customer, property: customer_id}, to: {type: Order, property: customer_ref}}
```

### 6.2 Semantics

- **Sources are names.** `kind` is `table` (relational rows), `graph` (a
  native property graph) or `document` (reserved, 12.3). How to reach a
  source is each consumer's own configuration, such as sql2graph's database
  settings.
- **Node bindings.** Each binds one model type to one source. A `table`
  binding names the table and the `id` columns that identify a row. A
  `graph` binding reads the source's nodes of that label. `properties`
  binds model properties to columns, optionally with a JSON `path` into a
  `json` column. Only declared properties may be bound, with a compatible
  type (4.6).
- **One type, several sources.** A type bound in more than one source must
  have a `key`; nodes with equal key values are one entity, as a
  migration's `MERGE` would make them.
- **Relation bindings** are one of three:
  - a `table` whose rows are the relationships, with `from.columns` and
    `to.columns` as foreign keys into each endpoint binding's `id`;
  - a `graph` source that stores the relationships itself;
  - a `join`, with no backing rows: the relationship is equality between a
    property on each endpoint.
- **Labels follow the model.** A mapping never declares labels. A bound
  node carries its type and its ancestors' labels (4.7).
- **Endpoints conform.** Every binding's endpoint types are admitted by the
  relation type's `start_labels` and `end_labels`.

### 6.3 Text is not mapped

Extraction from text needs no mapping file. Its source is the corpus, and
the model with its `extraction` profile is the whole binding.

### 6.4 Rendering

A mapping and its model render mechanically to a consumer's own format,
through a renderer that consumer supplies. For sql2graph's format today:

| hygm | sql2graph today |
|---|---|
| node binding, `table` | `NodeSource{type: table}` |
| property `column` / `path` | `PropertySource.field` |
| type plus ancestors | `GraphNode.labels` |
| relation binding, `table` | `RelationshipSource{type: table}` |

A rendering must be lossless for everything the target can express, and
must refuse what it can't, rather than drop it.

## 7. Layer 4: plans

A plan is what a target needs to hold a model: derived, deterministic,
idempotent. hygm computes it; it is not a file format.

### 7.1 Memgraph

| From | The plan adds |
|---|---|
| a type's `key` | a uniqueness constraint and a label-property index on the key |
| `required: true` | an existence constraint |
| a property's `type` | a data type constraint (`IS TYPED`) |
| `values` | an enum, and the property typed with it |
| `parents` | the ancestor labels every write sets |
| a multi-valued property | a `LIST` type constraint where Memgraph has one; no per-element type |
| `minimum`/`maximum`, `max_count`, `disjoint_with`, `value` | nothing: Memgraph can't enforce them; consumers that write data check them |
| `individuals` | `MERGE` statements seeding them, only when the caller asks |
| a mapping's `join` relation | an index on each endpoint's join property |

Target-specific additions that aren't implied by the model, such as a vector
index on an embedding property, belong in a `memgraph` profile, to be
registered when the first consumer needs one. This replaces sql2graph's
`GraphIndex`, `GraphConstraint` and `GraphEnum`.

## 8. Layer 5: producers

A producer makes a model, and optionally a mapping, from some input. Every
output passes the same gate (section 9) before anyone uses it.

| Producer | Input | Output | Where it is today |
|---|---|---|---|
| `ManualStrategy` | a model file | model | hygm |
| `OwlImportStrategy` | an OWL ontology | model (10) | hygm |
| `CatalogStrategy` | a SQL catalog: tables, columns, keys, foreign keys, in a documented source-neutral shape | model and mapping | sql2graph's deterministic strategy |
| `LlmRecommendationStrategy` | text sessions and a base model | model and its changes | hygm |
| `LlmSchemaStrategy` | a SQL catalog and a base model and mapping | model and mapping | sql2graph's LLM strategy |

Producers with heavy dependencies sit behind extras (`hygm[owl]`,
`hygm[llm]`, `hygm[sql]`), so the core stays dependency-light.

## 9. Validation

Every reader of a model checks, as critical errors:

1. The format: the JSON Schema of its version, unknown keys refused at every
   level (5.1 for profiles).
2. Labels unique among node types and among relation types; property names
   unique per type, counting inherited ones.
3. `parents` and every endpoint name declared node types; no inheritance
   cycle.
4. Property types are listed in 4.6; `values` only on `string`; redeclared
   and doubly inherited properties agree (4.7).
5. Every `key` property is declared or inherited and required; a type has
   at most one key.

A mapping reader also checks section 6.2. A profile's consumer also checks
that profile's rules.

Each error has a **code** from the table below and a message for people. Applying an operation (layer 6) to a model it doesn't fit reuses the layer-1 codes (`unknown_type`, `undeclared_property`, `duplicate_label`, `duplicate_property`).
Readers emit the same codes; their messages are their own. A file with
several problems may report any non-empty subset of them, so a conformance
fixture holds exactly one problem.

| Code | Layer | Meaning |
|---|---|---|
| `invalid_format` | any | Not a mapping, a required key missing, or a value of the wrong JSON type. |
| `unknown_version` | any | A `version` this reader doesn't know. |
| `unknown_key` | any | A key the format doesn't define, at any level. |
| `invalid_identifier` | 1 | A label, relation type or property name that isn't a valid identifier (4.10). |
| `duplicate_label` | 1 | Two node types, or two relation types, with one label. |
| `duplicate_property` | 1 | One property name declared twice on one type. |
| `unknown_parent` | 1 | `parents` names an undeclared node type. |
| `parent_cycle` | 1 | The parent graph has a cycle. |
| `undeclared_endpoint` | 1 | `start_labels` or `end_labels` names an undeclared node type. |
| `unknown_property_type` | 1 | A `type` not in 4.6. |
| `values_not_string` | 1 | `values` on a property whose type isn't `string`. |
| `property_conflict` | 1 | A redeclared or doubly inherited property disagrees on `type` or `values` (4.7). |
| `unknown_key_property` | 1 | `key` names a property the type neither declares nor inherits. |
| `key_not_required` | 1 | A key property isn't `required: true`. |
| `two_keys` | 1 | A type ends up with more than one key (4.8). |
| `unknown_profile` | 2 | A profile name that is neither registered nor `x_`-prefixed. |
| `invalid_profile_value` | 2 | A consumed profile's setting has the wrong type or value. |
| `profile_rule` | 2 | A consumed profile's own rule fails (such as 5.4's). |
| `unknown_type` | 1, 3 | A reference names a type the model doesn't declare: `disjoint_with`, `inverse_of`, a relation restriction, an individual's type or relation (1), or a binding (3). |
| `unknown_source` | 3 | A binding names a source the mapping doesn't declare. |
| `duplicate_source` | 3 | Two sources with one name. |
| `binding_kind` | 3 | A binding doesn't set exactly one of `table`, `graph`, `join`, or sets one its source's kind doesn't allow. |
| `undeclared_property` | 3 | A binding binds a property its type doesn't declare or inherit. |
| `unbound_key` | 3 | A type bound in more than one source has no `key`. |
| `endpoint_not_admitted` | 3 | A relation binding's endpoint types aren't admitted by its relation type. |
| `endpoint_columns` | 3 | A table relation's `from` or `to` columns don't match what they point at: the endpoint binding's `id` in the same source, else the endpoint type's `key`. |
| `unknown_model` | 3 | The mapping's `model` can't be read. |
| `import_cycle` | 1 | Imports form a cycle (4.12). |
| `disjoint_conflict` | 1 | Two disjoint types share a descendant, or a type is disjoint with an ancestor (4.13). |
| `duplicate_name` | 1 | A synonym equals another type's label or synonym (4.13). |
| `count_conflict` | 1 | `min_count` above `max_count`, `required` disagreeing with `min_count`, a multi-valued key, or an individual breaking a count (4.14, 4.17). |
| `invalid_value` | 1 | A `default`, `value`, bound or individual's value that isn't valid for its property (4.14, 4.17). |
| `relation_conflict` | 1 | A relation restriction that widens instead of narrowing (4.16). |
| `inverse_conflict` | 1 | Inverse relations whose endpoints aren't swapped, or a symmetric relation with different endpoints (4.16). |
| `duplicate_individual` | 1 | Two individuals with one `id` (4.17). |
| `abstract_instance` | 1 | An individual of an abstract type (4.17). |
| `missing_value` | 1 | An individual without a required property (4.17). |
| `unknown_individual` | 1 | A relation naming an individual that isn't declared (4.17). |
| `unknown_operation` | 6 | An operation hygm doesn't define. |
| `invalid_operation` | 6 | A rename or merge into itself, or an order list that doesn't name each item exactly once. |

### 9.1 Lint: the design guidelines

Validation (above) is the hard gate: a model that fails it is refused. Design
guidelines are judgement calls, so hygm checks them as **lint**. Lint only
reports, and never refuses a model or blocks a pipeline. Each finding has a code, a severity (`warning` or `info`), the
element it is about and a suggestion. `hygm lint` prints them, and
`lint_model(model)` returns them.

| Code | Severity | Finding |
|---|---|---|
| `single_subtype` | warning | A type with exactly one direct subtype: a modeling problem or an incomplete ontology. |
| `many_subtypes` | info | More than a dozen direct subtypes: intermediate categories may be missing. |
| `redundant_endpoint` | warning | An endpoint list naming a type and one of its descendants; drop the descendant. |
| `endpoint_covers_all_subtypes` | info | An endpoint list naming every subtype of a type but not the type: use the type. |
| `property_on_every_subtype` | info | Every direct subtype declares the same property (same type) and the parent doesn't: attach it to the parent. |
| `empty_subtype` | info | A subtype that adds no property, refinement or relation. Fine in a terminological hierarchy; worth a look otherwise. |
| `plural_of_parent` | warning | A type whose label is the plural (or singular) of an ancestor's (`Wine` under `Wines`). |
| `naming_case` | warning | A label that doesn't follow `ontology.conventions`. With no conventions, the majority style of the model is assumed. |
| `naming_number` | info | A type label that breaks the singular/plural convention. |
| `meta_word_in_name` | warning | A label containing `Class`, `Type`, `Property`, `Slot`, `Relation` or `Concept` as a word. |
| `sibling_naming` | info | Siblings where some repeat the parent's name and some don't (`RedWine` beside `White`). |
| `missing_description` | warning | A type, relation type or property with an empty description. |
| `no_competency_questions` | info | No `ontology.competency_questions`. |
| `unused_type` | info | A node type no relation names, no type inherits from and no individual has. |
| `inverse_not_declared` | info | Two relation types with swapped endpoints and names like `maker`/`produces` that aren't declared inverse. Heuristic. |

A consumer that wants a gate on top of lint builds it from the codes. hygm
never does.

## 10. OWL import

`OwlImportStrategy` extends its mechanical mapping:

| OWL / RDFS | hygm |
|---|---|
| `owl:Class` | node type |
| `owl:ObjectProperty` with `rdfs:domain` / `rdfs:range` | relation type with `start_labels` / `end_labels` (as today) |
| `rdfs:subClassOf` a named class | `parents` |
| `owl:DatatypeProperty` with `rdfs:domain` C | a property on C (on each class of an `owl:unionOf` domain); one with no domain is skipped |
| its `rdfs:range` | `type`, by 4.6 |
| an `owl:oneOf` of literals as range | `values` |
| `owl:hasKey` | `key`, each key property marked `required` |
| `owl:FunctionalProperty` with an `owl:minCardinality 1` restriction | `required: true` |
| each IRI | `iri` |
| `owl:Ontology` with `rdfs:comment`, `owl:versionInfo` | `ontology` (4.11) |
| `owl:imports` | `imports`, the imported ontology converted too |
| `owl:disjointWith`, `owl:AllDisjointClasses` | `disjoint_with` |
| `rdfs:label` with a language tag, `skos:prefLabel` | `names` |
| `skos:altLabel` | `synonyms` |
| `owl:inverseOf` | `inverse_of` |
| `owl:SymmetricProperty` | `symmetric` |
| `owl:FunctionalProperty` | `max_count: 1` |
| a cardinality restriction (`min`/`max`/exact, qualified or not) on a class | `min_count`/`max_count` on the property or relation restriction |
| `owl:hasValue` restriction | `value` |
| `owl:allValuesFrom` restriction on an object property | `relation_restrictions` end labels |
| `xsd:minInclusive`/`maxInclusive` datatype restriction | `minimum`/`maximum` |
| `owl:NamedIndividual` with its assertions | `individuals` |

**OWL export** writes the same table backwards, so a model round-trips
through OWL. `abstract` has no OWL counterpart and is written as an
annotation (`hygm:abstract`); `default`, `unit` and the profiles likewise.

Restrictions, equivalence, disjointness and other axioms stay out. hygm is
a schema, not a reasoner.

## 11. Layer 6: evolution

A model changes: a user edits it, derivation learns from sessions, a source
schema grows a column. Today sessions-graph's `Change` and sql2graph's
`ModelOperation`s say this in two vocabularies. hygm gets one:

- **Operations** are pure functions from a model to a model, with a JSON
  form: add, remove and rename a node type, relation type or property; set
  `parents`, `key`, endpoints, `description`, a profile setting; merge one
  type into another. Derivation's `retire`, `restore` and `veto` are
  operations plus a reason.
- **`diff(a, b)`** returns the operations that turn `a` into `b`.
- **A version** is a model, the version it came from, and the operations
  between them. sessions-graph's `(:OntologyVersion)` nodes are one store
  of versions.
- **Data migration is the consumer's.** hygm says what changed; moving the
  data is sessions-graph's or sql2graph's job.
- **Every key has an operation.** The keys of 4.11–4.17 (the `ontology`
  block, imports, `abstract`, `disjoint_with`, names, facets, relation facets
  and restrictions, individuals) each get a `set_*` or add/remove operation,
  so `diff` and `apply` cover the whole model.

Storing a version-2 model in a user's graph is a change older
sessions-graph clients would misread, since today they ignore unknown keys.
sessions-graph raises its graph epoch before it writes one (AGENTS.md,
"Graph epochs").

## 12. Open questions

### 12.1 Who owns source mappings (#695)

Other tools in the Memgraph ecosystem already have their own formats for
binding labels to sources. Either:

- **A. They read hygm mappings** as an alternative input. One mapping
  format everywhere; each tool's loader grows a second input.
- **B. They keep their own format**, which references a hygm model for its
  labels, keys, hierarchy and descriptions. hygm mappings are what Python
  producers (sql2graph, migrations) write, and they render losslessly into
  each tool's format (6.4).

Recommendation: **B now**. Everyone shares the ontology layer, which is
where the projects disagree today, without changing anyone's loader, and A
stays open once rendering has shown the formats are equivalent.

### 12.2 One file or two

This document keeps the model and its mappings in separate files, so one
model serves many mappings. A single file with a `mappings` section would
be simpler to hand around, at the cost of copying the model per source.

### 12.3 Documents

`document` sources (Mongo, JSON files) are reserved. Their binding (a
collection and a path per property) needs one real consumer before it is
specified.

### 12.4 Transformations

sql2graph's `PropertySource.transformation` lets a column be transformed on
the way in. This document allows only a column and a JSON path. Anything
more (casts, concatenation, expressions) means picking an expression
language every reader implements.

## 13. Contract artifacts

- **JSON Schemas** in `hygm/src/hygm/schema/`, shipped in the wheel:
  `model.v1.schema.json`, `model.v2.schema.json`, `mapping.v1.schema.json`,
  and one per registered profile. Rules a JSON Schema can't express are
  defined by this document.
- **Conformance fixtures** in `hygm/conformance/{model,mapping}/`: every
  file under `valid/` must load, and every file under `invalid/<code>/`
  must be refused with that error code. Each invalid fixture holds exactly
  one problem. hygm's tests run all of them, and a reader in another
  language runs the same files, so a change on one side fails the other
  side's CI.
- **Versioning.** Adding an optional key, a property type or a profile is
  additive: it keeps the format version, but every reader learns it before
  any producer writes it. Changing what an existing file means needs a new
  format version.
  The keys of 4.11–4.17 are additive in this sense: they stay in version 2,
  and a version-2 reader from before them refuses a file that uses them
  (unknown keys) rather than misreading it.

## 14. Python package

```
hygm/
  model/        layer 1: NodeType, RelationType, Property, HygmModel, I/O, validation
  profiles/     layer 2: extraction, conversation, learning
  mapping/      layer 3: Mapping, bindings, I/O, validation, rendering
  plan/         layer 4: memgraph
  strategies/   layer 5: manual, owl (import and export), catalog, llm
  lint/         the design guidelines (9.1)
  evolution/    layer 6: operations, diff, versions
  schema/       JSON Schemas
```

`HygmModel` gains `ancestors(label)`, `descendants(label)`,
`is_a(label, other)`, `properties_of(label)` (including inherited) and
`key_of(label)`. `endpoint_labels` expands each label to it and its
descendants; with no `parents` anywhere it returns what it returns today.
Today's top-level names stay importable from `hygm` for one minor version,
then move.

## 15. Plan

1. **This document**, reviewed and agreed, including 12.1.
2. **Layers 1 and 2:** version 2, version-1 reading, properties, `parents`,
   `key`, `values`, `iri`, the three profiles, error codes, JSON Schemas,
   fixtures. `validate_model` drops the `User`/`Person` rule into
   `conversation`; the default model declares its profiles; readers that
   don't yet implement a feature refuse models that use it
   (unstructured2graph: `parents`; derivation: `parents`, `properties`,
   `key`). hygm 0.3.0.
3. **Layers 3 and 4:** mappings, rendering, the Memgraph plan.
4. **sql2graph onto hygm,** in steps: it emits a hygm model and mapping
   beside its current output; `CatalogStrategy` and `LlmSchemaStrategy`
   move into hygm; its operations become hygm operations; then
   `agents/sql2graph/core/hygm` is deleted. Its per-element statistics
   (`count`, `filling_factor`, `examples`) describe a dataset, not a model,
   and stay in sql2graph.
5. **OWL import** extension (section 10).
6. **Evolution:** one operation vocabulary for derivation and sql2graph.
7. **Other ecosystem tools** read the model, then mappings if 12.1 lands
   on A.
8. **Later, by demand:** inheritance in extraction, hierarchy-aware
   derivation, list properties, documents, OWL export.

## 16. Out of scope

- Relation inheritance (`rdfs:subPropertyOf`), relation keys, cardinality.
- Reasoning of any kind.
- Connection details and credentials (principle 7).
