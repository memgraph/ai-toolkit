# What `gliner2.joint_ie.JointSchema` actually enforces

Research note for [#345](https://github.com/memgraph/ai-toolkit/issues/345) (part of #344).
Answers the six questions on that ticket about whether `JointSchema` is a viable
replacement for `GLiNER2Backend._extract_sync`'s current `model.create_schema()` path.

**Status: findings, not a decision.** Nothing in `unstructured2graph` was changed.

## How this was established

Primary sources only:

- **Library source** read on disk, `gliner2==2.0.0` (latest on PyPI), installed into an
  isolated venv at
  `/private/tmp/claude-501/-Users-antejavor-repos-ai-toolkit/26506fb8-e763-4d7c-a1c5-41d42291a42b/scratchpad/gliner2-venv`.
  All source citations below are `<venv>/lib/python3.13/site-packages/gliner2/<path>:<line>`
  and are abbreviated to just `<path>:<line>`.
- **Running it**, same venv, checkpoint `fastino/gliner2.5-base-v1`. Scripts live in
  that venv's sibling `scratchpad/g2/` directory; every empirical claim below quotes
  the actual output.
- **Upstream repo** <https://github.com/fastino-ai/GLiNER2>, specifically
  `tutorial/15-joint_ie.md` (fetched via `gh api repos/fastino-ai/GLiNER2/contents/tutorial/15-joint_ie.md`).
  The installed wheel does **not** ship the tutorials, so this file is not discoverable
  from the installed package — which is most of why the richer API went unnoticed.

One environment note, because it cost time: the venv could not load the checkpoint until
`protobuf` and `sentencepiece` were installed (`AutoTokenizer` needs them for the
DeBERTa-v2 tokenizer). `gliner2[local]` does not pull them in.

> The checkpoint matters. `fastino/gliner2.5-base-v1` is a **boundary** architecture model
> (`AutoExtractor.from_pretrained(...).architecture == 'boundary'`, verified). GLiNER2 has
> two scoring paths — dense span (`joint_ie/candidates.py`) and sparse boundary
> (`joint_ie/candidate_scores.py`) — and they enforce typing at *different* places.
> Everything empirical below is the boundary path. Where the span path differs, it is
> called out.

---

## Q1 — Entry point

**A `JointSchema` does not satisfy `AutoExtractor`'s `extract()` at all. It needs a
different object: `gliner2.joint_ie.JointIEEngine` (alias `JointIE`).**

`AutoExtractor.from_pretrained()` returns a `SpanExtractor`/`BoundaryExtractor`
(`auto.py:124-130`), whose `extract()` comes from `ExtractorRuntimeMixin`
(`inference/runtime.py:1159-1168`) and whose `create_schema()` returns the *old*
`gliner2.inference.schema.Schema` (`inference/runtime.py:61-63`) — the one whose
`.relations()` hard-codes `{"head": "", "tail": ""}` (`inference/schema.py:448`).

Four call shapes tried against a loaded model (`scratchpad/g2/q1_entry.py`):

| call | result |
| --- | --- |
| `model.extract(text, JointSchema)` | `AttributeError: 'JointSchema' object has no attribute 'get'` at `inference/runtime.py:189` |
| `model.extract(text, JointSchema.to_dict())` | **returns `{}`, silently** — no error, no output |
| `model.extract(text, compile_schema(s).build())` | works, but degrades to the old API (see below) |
| `JointIEEngine(model).extract(text, JointSchema)` | works, returns a `JointResult` |

The silent `{}` is worth flagging: `to_dict()` emits `relations` as a *mapping*
(`joint_ie/schema.py:115`) while the legacy runtime expects a *list*
(`inference/runtime.py:187-207`), and the mismatch produces an empty result rather than
an error.

`compile_schema(s).build()` "works" only in the sense that it runs — `compiler.py:26-28`
rebuilds the model-facing schema as `{n: {"head": "", "tail": ""}}`, i.e. **all head/tail
typing is dropped** and the run is equivalent to what the backend does today. The typing
survives only on the `CompiledJointSchema.relation_specs` / `.constraints` fields, which
only `JointIEEngine` reads.

### The working call

```python
from gliner2.joint_ie import JointIE, JointIEConfig   # JointIE is JointIEEngine

joint = JointIE.from_pretrained("fastino/gliner2.5-base-v1")   # loads via AutoExtractor
# or: JointIEEngine(already_loaded_autoextractor_model)

schema = (joint.create_schema()                                 # -> JointSchema
          .entities(["person", "organization", "location"])
          .relation("works_for", "person", "organization")
          .relation("located_in", "organization", "location"))

result = joint.extract(text, schema, config=JointIEConfig(optimizer="beam", beam_size=32))
```

Verified output (`scratchpad/g2/q1b_callpath.py`, text
`"Alice works for Acme in Paris. Bob joined Acme last year."`):

```
JointIE.from_pretrained -> JointIEEngine | wraps BoundaryExtractor | arch: boundary
create_schema() -> JointSchema
extract(): {'entities': [{'id': 'e1', 'type': 'person', 'text': 'Alice', 'start': 0, 'end': 5,
            'confidence': 0.998...}, ...],
            'relations': [{'type': 'works_for', 'head': 'e1', 'tail': 'e2',
                           'confidence': 0.985...}, ...]}
```

Notes on the entry point:

- `JointIEEngine.__init__(model)` accepts an already-loaded `AutoExtractor` model
  (`joint_ie/engine.py:134-151`), so the backend's existing `model` constructor argument
  and its "pre-loaded model or test fake" contract can be kept.
- `JointIEEngine.extract()` accepts **only** a `JointSchema` or an already-compiled one.
  A plain dict raises `TypeError: schema must be a JointSchema` (`joint_ie/compiler.py:23`,
  reproduced). `JointSchema.from_dict()` exists (`joint_ie/schema.py:117-132`) for
  round-tripping config.
- Prediction knobs (thresholds, beam size, `include_spans`, `include_confidence`) live on
  `JointIEConfig` per call, never on `from_pretrained` — `from_pretrained` actively
  rejects them (`joint_ie/engine.py:156-166`).
- `batch_extract(texts, schema | [schemas])` also exists and works
  (`joint_ie/engine.py:365-381`, verified).

---

## Q2 — Long text: `extract_long()` composes, but constraints are **per-window only**

`JointIEEngine.extract_long(text, schema, *, config, chunk_size=384, chunk_overlap=64)`
exists (`joint_ie/engine.py:383-390`) and delegates to
`joint_ie/long_text.py::extract_long_text`. It works and remaps spans to document
offsets, same as the legacy `extract_long()`.

**Signature difference that will bite the port:** `include_spans` / `include_confidence`
are *not* kwargs here; they come from `config`. Verified:

```
extract_long(..., include_spans=True, include_confidence=True)?
  TypeError: JointIEEngine.extract_long() got an unexpected keyword argument 'include_spans'
```

### Cross-window relations: never produced

Each chunk is extracted independently (`long_text.py:32-34`) and the merge only keeps a
relation when **both** endpoint ids came from the *same* chunk result
(`long_text.py:51-56`):

```python
# Both IDs must originate in this same chunk result. This explicit
# lookup is what prevents accidental cross-window edge creation.
if relation.head not in local_keys or relation.tail not in local_keys:
    continue
```

The module docstring states it outright (`long_text.py:21-22`): *"Consequently this
function never synthesizes cross-chunk relations."* Upstream's tutorial says the same
("A relation is kept only if **both endpoints were extracted in the same chunk**.
Cross-window edges are never synthesized.").

Empirically (`scratchpad/g2/q2356.py`), a 670-char text with `Alice Johnson` only in
window 0 and `Memgraph` only in windows 2–3, `chunk_size=40, chunk_overlap=8`:

```
--- extract_long, Alice in window 0 / Memgraph in a later window:
    feasible=True entities=3 relations=0
```

So **the window-1/window-3 case the ticket asks about yields no edge, ever.** This is not
a JointSchema regression — it is exactly what the current backend already suffers
(`_entity_id`'s "no cross-chunk coreference" note is the same class of limitation) — but
it means `JointSchema` buys nothing for facts whose endpoints are far apart.

### Constraints do **not** hold across the merge

`long_text.py` never touches `schema.constraints`, never re-runs an optimizer, and simply
unions relation rows keyed by `(type, head_key, tail_key)` (`long_text.py:36-77`). Two
independent demonstrations:

**1. Real model, natural text, no contrivance** (`scratchpad/g2/q2e_overlap.py`). Every
compiled `JointSchema` carries `EntityOverlapPolicy(policy='disallow')` by default
(`compiler.py:30-32`), which forbids *any* overlapping entity spans:

```
extract()         : 6 entities, overlap violations = 0
extract_long(12/8): 9 entities, overlap violations = 3
    ('city', 'Fastino', (119, 126), 'company', 'Fastino', (119, 126))
    ('city', 'Fastino', (119, 126), 'person',  'Fastino', (119, 126))
    ('company','Fastino',(119, 126), 'person', 'Fastino', (119, 126))
```

The same span is simultaneously a `city`, a `company` and a `person` in the merged
result. The single-window decoder never allows that; the merge produced it.

**2. Controlled, on the library's own merge function** (`scratchpad/g2/q2d_crosswindow.py`).
A stub engine feeds `extract_long_text` two per-chunk results that *each individually*
satisfy `max_per_head=1`, with one `Alice Johnson` at identical document offsets in both
windows:

```
schema constraints include MaxRelationsPerHead(limit=1, relation='works_for')
relations: [('works_for', 'Alice Johnson', 'Memgraph'), ('works_for', 'Alice Johnson', 'Fastino')]
relations per head id: {'e2': 2} -> max_per_head=1 HELD globally? False
```

**Bottom line for Q2: constraints are guaranteed only within one window.** Across windows
you get set-union semantics with no re-check. For a design that wants document-level
cardinality (e.g. "a session has one owner"), that guarantee has to be re-imposed on our
side after `extract_long`.

*Not determined:* whether re-running the optimizer over the merged graph is feasible as a
cheap post-pass. The constraint objects are decoder-independent and duck-typed
(`constraints.py:1-12`, `Constraint.apply` at `constraints.py:83-88` takes a plain
iterable of relation-like objects), so it looks mechanically possible, but `Constraint.apply`
has **zero call sites in the package** (verified by grep) and is therefore untested-in-anger
library surface.

---

## Q3 — Output shape: completely different

Not `{"entities": {...}, "relation_extraction": {...}}`. `JointIEEngine.extract()` and
`.extract_long()` both return a `JointResult` dataclass (`joint_ie/result.py:66-164`),
whose `.to_dict()` is:

```json
{
  "entities": [
    {"id": "e1", "type": "person", "text": "Alice Johnson", "start": 0, "end": 13,
     "confidence": 0.998}
  ],
  "relations": [
    {"type": "works_for", "head": "e1", "tail": "e2", "confidence": 0.985}
  ]
}
```

Differences that matter for `_extract_sync`:

| today (`inference.Schema`) | `JointResult` |
| --- | --- |
| `entities` is a dict keyed by entity type, values are span lists | `entities` is a **flat list**; the type is a field |
| relations under key `relation_extraction` | key is `relations` |
| `head`/`tail` are **span dicts** (`{"text","start","end","confidence"}`) | `head`/`tail` are **entity ids** (`"e1"`), stable within one result |
| relation carries no confidence of its own (backend docstring says so; it reuses head/tail confidence) | relation carries its **own** `confidence` |
| head/tail must be matched back to entities by span/text | already resolved — `result.entity(rel.head)` |

The whole `_resolve_entity_id` / `entity_id_by_span` / `entity_id_by_text` matching
apparatus in `gliner2_backend.py` becomes unnecessary.

`include_spans` and `include_confidence` still work, via `JointIEConfig`, and they behave
as pure *serialisation* flags — verified all four combinations
(`scratchpad/g2/q2356.py`). With both `False`:

```json
{"entities": [{"id": "e1", "type": "person", "text": "Alice Johnson"}],
 "relations": [{"type": "works_for", "head": "e1", "tail": "e2"}]}
```

but the underlying dataclass still carries the offsets:

```
raw JointEntity still carries start/end:
  JointEntity(id='e1', type='person', text='Alice Johnson', start=0, end=13,
              confidence=None, sentence_id=None, rescued=False)
```

Extra fields on the result worth knowing: `JointResult.feasible`
(`result.py:73-75`), `JointEntity.rescued` (`result.py:21`), `JointRelation.derived`
(`result.py:51`), plus graph helpers `outgoing` / `incoming` / `neighbors` /
`relations_of` / `to_networkx`.

---

## Q4 — Real enforcement vs post-filtering (the important one)

**Verdict: genuinely applied during decoding, not as a post-hoc filter.** Stronger than
that — the constraints split across *two* stages, and head/tail typing plus `allow_self`
are enforced even earlier than the beam search, inside the model's pair generator.

Be warned that `joint_ie/constraints.py` opens with the line
`"""Post-decoding constraints for joint entity and relation extraction.` — that docstring
is misleading about *when* the constraints run. The call sites say otherwise.

### Stage 1 — typed endpoints and self-loops: enforced before scoring

On the boundary path, a relation edge candidate is only ever constructed for a
head type drawn from `relation_spec.head` and a tail type from `relation_spec.tail`
(`joint_ie/scoring.py:621-647`):

```python
for head_type in getattr(relation_spec, "head", ()):
    head_key = (head_type, *head_span)
    if head_key not in mention_keys:
        continue
    for tail_type in getattr(relation_spec, "tail", ()):
        ...
        edges.append(ScoredRelationEdge(relation_type=relation_type, head=head_key, tail=tail_key, ...))
```

and `allow_self` is pushed *into the model's tensorised pair generator*
(`joint_ie/scoring.py:439-452` builds a `RelationTypeSpec(..., allow_self=...)`, consumed
at `models/boundary/relations.py:217`):

```python
pair_valid &= allow_self[..., None, None] | ~same_span
```

Head/tail query routing is likewise per relation type
(`models/boundary/relations.py:147-149`).

Measured on the candidate set *before the optimizer runs*
(`scratchpad/g2/q4b_typing.py`):

```
works_for person->company
  CandidateScoreSet.edges = 4 scored relation candidates
  endpoint TYPE pairs present among candidates: [('person', 'company')]
works_for city->person  (deliberately wrong types)
  CandidateScoreSet.edges = 4
  endpoint TYPE pairs present among candidates: [('city', 'person')]
works_for [person,company,city]->[person,company,city]
  CandidateScoreSet.edges = 36
  endpoint TYPE pairs: [('city','city'), ('city','company'), ..., ('person','person')]

  allow_self=False: 16 candidates, 0 of them self-pairs
  allow_self=True : 24 candidates, 8 of them self-pairs
```

A mistyped pair is never scored. A self-pair is never scored unless `allow_self=True`.
This is as literal as "invalid combinations are never admitted into the search" gets.

*Span-architecture note:* the dense path reaches the same result differently —
`joint_ie/candidates.py:332-348` builds typed role candidates from
`hypothesis.head_types` / `tail_types` at candidate-generation time. Not exercised here,
since our checkpoint is boundary.

### Stage 2 — cardinality and graph constraints: enforced during beam expansion

`BeamOptimizer.optimize` consults every constraint *before* an expansion is added to the
beam (`joint_ie/optimizers/beam.py:55-76`):

```python
for state in beam:
    ...
    if not all(self.allow_node(problem, node, proposed_nodes, state.edges) for node in added_nodes):
        continue
    if not self.allow_edge(problem, edge, proposed_nodes, state.edges):
        continue
    ...
    expanded.append(_State(...))
```

Same structure in `GreedyOptimizer.optimize` (`optimizers/greedy.py:43-51`). Only *after*
the search does `validate_solution` (`optimizers/base.py:70-101`) re-check the final
assignment, and it exists specifically to catch *derived companion edges* injected for
symmetric/inverse relations, which never went through the incremental screen
(`base.py:74-79`, `base.py:130-165`).

#### Empirical discriminator

The test that separates the two hypotheses is *when* the rejections happen. I patched
`BaseOptimizer.allow_edge` to count rejections, with a phase flag flipped by a patched
`validate_solution` (`scratchpad/g2/q4_enforcement.py`), on
`"Alice Johnson works for Memgraph in Zagreb. Bob Smith works for Fastino in Paris."`:

```
unconstrained : allow_edge calls=302  rejected during beam search=0   rejected during final validate=0
                -> 4 works_for edges (both true pairs plus both cross pairs)

max_per_head=1: allow_edge calls=250  rejected during beam search=26  rejected during final validate=0
                -> 2 works_for edges: Alice->Memgraph, Bob->Fastino
```

**26 rejections during the search, 0 at final validation.** A post-filter would show the
mirror image.

Second, independent discriminator — the feasibility signal. If constraints were only
checked post-hoc, `validate_solution` would fail on the natural output and
`BeamOptimizer` would log a warning and return the **empty** solution with
`feasible=False` (`optimizers/beam.py:96-109`). Observed instead:

```
unconstrained      feasible=True  works_for=4  all_rels=6  entities=6
max_per_head=1     feasible=True  works_for=2  all_rels=4  entities=6
max_per_tail=1     feasible=True  works_for=2  all_rels=4  entities=6
max_per_head=0     feasible=True  works_for=0  all_rels=2  entities=6
```

Reduced-but-non-empty and `feasible=True` is only reachable by incremental screening.
(`max_per_head=0` is accepted by the schema — `schema.py:56-58` only requires
non-negative — and cleanly suppresses a relation type entirely.)

### One limit on "valid by construction" worth recording

Blocking an edge does **not** remove its endpoints from the entity output. Both optimizers
run a second, edge-independent pass that admits any node with positive standalone score
(`beam.py:39-45`, `greedy.py:58-64`). Verified: with `max_per_head=0` and entity
thresholds pushed high enough that the four entities only entered via relation-endpoint
rescue, all four still appear in the result with `rescued=True` while the relations are
gone (`scratchpad/g2/q4_enforcement.py`, probe P3). So constraints govern the **edge
set**, not membership of the node set.

---

## Q5 — Unconstrained relations: **not expressible as empty; use "all declared types"**

`head` and `tail` are required positional parameters (`joint_ie/schema.py:78`) and
`_types()` rejects every empty form (`joint_ie/schema.py:12-17`). Verified
(`scratchpad/g2/q2356.py`):

```
FAIL relation('r') with head/tail omitted : TypeError: JointSchema.relation() missing 2 required
                                            positional arguments: 'head' and 'tail'
FAIL head=''       : ValueError: relation head contains an invalid entity type
FAIL head=()       : ValueError: relation head must contain at least one entity type
FAIL head=None     : TypeError: 'NoneType' object is not iterable
FAIL head='ghost'  : ValueError: relation 'r' references unknown entity types: ['ghost']
FAIL from_dict({"relations": {"r": {}}}) : TypeError: ... missing 2 required positional arguments
OK   head=['person','company'] tail=['person','company']
```

So the ticket's backward-compat rule — `start_labels=()` / `end_labels=()` means
*unconstrained*, never a validation failure — **cannot be satisfied by passing an empty
tuple through to `JointSchema`.** Our layer has to translate empty to
"every declared entity type".

That translation is sound. Both `head` and `tail` accept a list
(`joint_ie/schema.py:12-17`), and the resulting `TypedEndpoints` constraint admits
anything in the set (`constraints.py:114-118`). Verified end to end:

```
compiled: TypedEndpoints(relation='related_to', head_types=('person','company','city'),
                                                tail_types=('person','company','city'))
candidates: 36 edges spanning all 9 ordered type pairs, including ('person','person')
extracted: REL related_to 'Alice Johnson' -> 'Memgraph'
```

Two residual caveats:

1. "Anything goes" means "any *declared* entity type". There is no way to say "any span,
   typed or not" — relations are closed over the entity vocabulary by construction
   (`joint_ie/schema.py:90-91`).
2. Cost is quadratic in the type count at candidate time (3 types → 36 candidates vs 4 for
   one typed pair, measured above). With a large ontology, blanket-unconstrained relations
   will be noticeably more expensive.

Upstream's own answer to this question is to not use `JointSchema` at all:
"For unconstrained relation tuples (no typed endpoints, no uniqueness), see
[Relation Extraction](6-relation_extraction.md)" — i.e. the legacy
`inference.Schema` path the backend uses today.

Also note `compiler.py:33-39` attaches `NoSelfLoops`, `UniqueRelationPair` and
`UniqueRelationSlot` to **every** relation by default, so even an "unconstrained" relation
is not fully unconstrained: `allow_self=True` is needed to permit reflexive edges, and a
given ordered `(head, tail)` pair can appear at most once per relation type.

---

## Q6 — Entity/relation coupling: **guaranteed. The skip-and-log path becomes dead code.**

Yes — with `JointSchema` + `JointIEEngine`, a relation endpoint is always present in the
entity output. This is enforced at four independent layers:

1. **Scoring.** Relation endpoint spans are *force-scored* as entity mentions for every
   declared head/tail type and appended as `extra_mentions`
   (`joint_ie/scoring.py:509-533` collects `endpoint_requirements`,
   `scoring.py:569-597` scores and emits them).
2. **Candidate assembly.** Those endpoint mentions bypass the mention threshold *and* the
   per-type cap via `rescue_ids` (`joint_ie/candidate_scores.py:283-312`), and any edge
   whose endpoint did not survive is dropped rather than kept dangling
   (`candidate_scores.py:344-347`). The dense span path does the same with
   `CandidateSource.RELATION_RESCUE` (`joint_ie/candidates.py:339-347`).
3. **Problem invariant.** `JointProblem.__post_init__` raises
   `"edge endpoints must refer to nodes in the problem"` (`joint_ie/candidates.py:184-191`).
4. **Result invariant.** `ResultBuilder._endpoint_id` raises
   `"relation endpoint does not identify a selected entity"` (`joint_ie/result.py:292-295`),
   and `JointResult.__post_init__` raises
   `"relation endpoints must reference entities in this result"` (`joint_ie/result.py:83-85`).
   Reproduced directly.

The exact `located_in` / townhouse / Brookside case from
`gliner2_backend.py::_extract_sync`'s docstring, re-run under `JointSchema`
(`scratchpad/g2/q2356.py`):

```
--- located_in building->neighborhood: feasible=True entities=3 relations=1
    ENT e1 'building'     'townhouse'     [14,23) conf=0.996 rescued=False
    ENT e2 'neighborhood' 'Brookside'     [31,40) conf=0.990 rescued=False
    ENT e3 'person'       'Alice Johnson' [71,84) conf=0.997 rescued=False
    REL located_in 'townhouse' -> 'Brookside' conf=0.792
```

Both endpoints surfaced as entities. The coupling holds in the other direction too: when
entity thresholds were pushed to 0.9999 so no endpoint could be admitted as an entity, the
**relation disappeared with it** rather than being emitted dangling
(`entities=0 relations=0`).

So `aingest_chunk`'s `head_id is None or tail_id is None` skip-and-log branch
(`gliner2_backend.py:368-373`) is unreachable under this API — as is the whole
span/text matching that feeds it, since `head`/`tail` arrive as entity ids (Q3).

One caveat, straight from Q2: this guarantee is a property of `JointResult`, and
`extract_long_text` rebuilds a `JointResult` from merged fragments
(`long_text.py:63-77`). It keeps the invariant by *dropping* any relation whose endpoints
did not both come from the same chunk (`long_text.py:51-56`), so the result is still
internally consistent — you just silently lose those edges, with no log line.

---

## Summary table

| # | Question | Answer |
| --- | --- | --- |
| 1 | Entry point | Not `AutoExtractor.extract`. Use `gliner2.joint_ie.JointIEEngine` / `JointIE`; it wraps an existing model. A dict is rejected; `JointSchema` only. |
| 2 | Long text | `extract_long()` exists and composes. Constraints hold **within a window only**; cross-window relations are never synthesized and the merge does no re-check. |
| 3 | Output shape | `JointResult`, not `{"entities", "relation_extraction"}`. Flat entity list, `relations` with entity-id endpoints and own confidence. `include_spans`/`include_confidence` move to `JointIEConfig`. |
| 4 | Enforcement | **Real.** Typing + `allow_self` before scoring; cardinality/graph constraints during beam expansion (26 in-search rejections vs 0 at validate). Caveat: constraints govern edges, not node membership. |
| 5 | Unconstrained | **Not expressible as empty** — every empty form raises. Must be translated to "all declared entity types"; that works, at quadratic candidate cost. |
| 6 | Coupling | **Guaranteed** at four layers. The skip-and-log path becomes dead code. |

## Things I could not determine

- Whether re-running the optimizer over a *merged* multi-window graph is practical as a
  post-pass to restore document-level constraints. The machinery looks reusable
  (`Constraint` is duck-typed over plain mappings) but `Constraint.apply` has no call
  sites anywhere in the package, so it is untested library surface.
- Whether the span (non-boundary) architecture behaves identically. It is a genuinely
  different candidate path (`joint_ie/candidates.py` vs `joint_ie/candidate_scores.py`),
  and on that path `UniqueRelationSlot(..., slot='slot')` keys on `count_slot`
  (`candidates.py:363`), which is bounded by `JointIEConfig.count_top_k` (default 2) —
  that would cap relations per type per window at 2. On the boundary path `slot` is a
  unique per-edge index (`candidate_scores.py:359`) so the constraint is inert. **Not
  verified on a span checkpoint**; it only matters if we ever switch checkpoints.
- Real-world quality impact. Everything here is mechanism, measured on short synthetic
  sentences. Whether `JointSchema` actually raises edge yield/precision on our session
  corpus needs the existing extraction-quality eval (#332) re-run.
