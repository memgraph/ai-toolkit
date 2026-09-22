# What `gliner2==2.0.0` actually offers for entity attributes

Research note for [#362](https://github.com/memgraph/ai-toolkit/issues/362) (part of #344).
Answers the six questions on that ticket. Consumer is
[#361](https://github.com/memgraph/ai-toolkit/issues/361) — "Values, not links: do
entity attributes belong in the model".

**Status: findings, not a decision.** Nothing in `unstructured2graph` was changed.

Prior art: [the #345 note](https://github.com/memgraph/ai-toolkit/blob/research/gliner2-jointschema/unstructured2graph/docs/research/gliner2-jointschema.md)
(`JointIE` entry point, `JointResult` shape, constraint enforcement) and
[#350's resolution comment](https://github.com/memgraph/ai-toolkit/issues/350)
(3 of 6 question types had no expressible answer as a relation). This note assumes both.

## How this was established

Primary sources only. No upstream blog, README or tutorial was consulted — upstream
markets "span attributes" for 2.5, and establishing what the *installed package* and
*the checkpoint we run* do was half the job.

- **Library source on disk**, `gliner2==2.0.0`, in the only venv that has it:
  `/Users/antejavor/repos/ai-toolkit/.claude/worktrees/backend-comparison/.venv`.
  All `file:line` citations are relative to
  `<venv>/lib/python3.11/site-packages/gliner2/` and abbreviated to `<path>:<line>`.
- **Live runs** against `fastino/gliner2.5-base-v1` (boundary architecture, verified),
  from that same venv. Probe scripts live in this session's scratchpad at
  `.../scratchpad/g2attr/` (`q1_joint_surface.py`, `q23_legacy.py`,
  `q245_joint_compose.py`, `q34_constraints.py`, `q56_windows_alt.py`,
  `q25_tradeoff.py`); every empirical claim below quotes the actual output. Each
  script is self-contained apart from `fixture.py`, reproduced in full at the end.
- **Package metadata** (`gliner2-2.0.0.dist-info/METADATA`) and the PyPI version
  index, for Q6.

The test text is a single 572-char reconciled-session-shaped window: turns prefixed
`user: ` / `assistant: ` joined by blank lines, carrying the exact fact shapes #350
found inexpressible — a race time (`25:50`), a count (`3`), a shift window
(`8am-4pm`) — plus a price (`$129.99`) and a date (`March 14, 2025`). It is quoted in
full at the end.

---

## Short answers

**Q1 — Is there an attribute mechanism on the joint path at all? No. None, anywhere.**
`EntitySpec` has exactly six fields and not one of them holds a value
(`joint_ie/schema.py:22-32`). `JointSchema.entity()` is keyword-only with no
`**aliases` escape hatch (`joint_ie/schema.py:68-70`), so every smuggling attempt is a
`TypeError`. `JointSchema` has no `field`, `structure`, `classification` or
`entity_attributes` method. `compile_schema` hard-codes `"json_structures": []` and
`"classifications": []` into the model-facing schema (`joint_ie/compiler.py:26-28`), and
`JointResult`/`JointEntity` have no slot a value could land in
(`joint_ie/result.py:11-20`, `66-74`). Smuggling a structure *into* `model_schema`
by hand runs without error and is silently dropped from the result. The joint path is
entities and relations, exclusively.

**Q2 — The legacy API is the real mechanism, and there are three of them, not one.**
`gliner2.inference.schema.Schema` offers (a) `entity_attributes()` with
`AttributeGroup` — closed label sets attached to entity *spans*
(`inference/schema.py:59-76`, `325-407`); (b) `structure(name).field(...)` — value
fields that return real text spans with offsets (`inference/schema.py:103-135`); and
(c) `classification()` — document-level closed label sets. All three compose with
entities *and* relations in a **single** `extract()` call — verified, result keys
`['facts', 'entities', 'topic', 'relation_extraction']`. But that single call is the
*legacy* runtime, which is exactly what #345 retired: its relations compile to
`{"head": "", "tail": ""}` (`inference/schema.py:448`, confirmed in the built schema)
and their endpoints come back as span dicts needing re-matching. **The trade is
therefore: values in the same pass, or typed relations — not both.** Keeping
`JointSchema` means a second inference call per window, and that second call has no
entity ids at all, so the span-matching apparatus #345 retired comes back.

**Q3 — Three different landing places, and only one of them is a real value.**
An `AttributeGroup` value is a **label from a closed set** — `{"label": ..., "confidence": ...}`,
no offsets, no grounding in the text (`models/boundary/engine.py:469-485`). It *is*
co-located with its entity span inside the legacy entity dict, but the legacy output
has no entity ids, so coupling to a `JointResult` entity still needs span matching. A
`structure` field value is a **real span**: `{"text": "25:50", "confidence": 0.999,
"start": 82, "end": 87}` (`models/boundary/engine.py:762-792`) — but it is
**document-scoped, not entity-attached**: one instance per structure per document, with
no link to any entity. Record mode (`structure(..., mode="natural", anchor=...)`,
`inference/schema.py:246-265`) is the only construct that gives *per-instance* values
with an anchor span, and its anchor span does match joint entity offsets exactly — but
its field-to-anchor binding was wrong in the one case tested. `classification` is
document-level and entity-free.

**Q4 — Values are constrained by construction, but nothing is enforced during decoding,
and there is no feasibility signal.** Three kinds of constraint exist and they differ in
when they bite. A closed label set (`AttributeGroup.labels`, or a field's `choices=`) is
enforced *by construction*: the model only ever scores the declared labels. A type
constraint is a `RegexValidator`, and it is a **pure post-hoc filter** — with and without
a matching validator the same span comes back at byte-identical confidence
(0.9882827401161194), proving the decode was unchanged; a non-matching validator just
deletes the result (`models/boundary/engine.py:561-580`). Cardinality is `dtype`:
`"str"` takes `spans[0]`, `"list"` takes all (`models/boundary/engine.py:769-778`).
There is **no `allow_edge`-equivalent and no `feasible`-equivalent.** The legacy result
is a plain `dict` with no `feasible` field. Worse in two directions: a single-label
attribute group **can never abstain** — softmax argmax over the declared labels always
returns one (`models/boundary/engine.py:479-485`), verified by every entity in the text
being assigned `planet: "Mars"`; and a `choices=` field **hallucinates** — it returned
`"alpha"` at confidence 0.99 from a choice list none of whose members appears anywhere
in the text. The only abstention route in the whole surface is `multi_label=True` plus a
threshold. Unfillable fields come back `null`, and a structure whose fields are *all*
empty is dropped from the result entirely and silently
(`models/boundary/engine.py:590-591`).

**Q5 — It composes with per-window driving, cheaply, but document-scoped fields
conflict across windows and there is no signal saying so.** Per-window driving works
(we call `extract()` per window; offsets are window-local and need the `+offset` remap
we already do). Cost is modest: one attribute group of 3 labels adds ~12ms to a 123ms
window (+10%); a 6-field structure adds ~12ms (+10%); an 18-field structure adds ~88ms
(+72%). Candidate expansion is linear, not quadratic — attribute labels are injected as
extra *entity* queries (5 → 8 for 3 labels, `inference/schema.py:400-402`), structure
fields as extra field queries. The real cost is the second pass: joint-then-legacy is
299ms vs 157ms for joint alone, i.e. **+90% wall clock per window**. The composition
problem is worse than the cost: a structure is one instance per *document*, so cutting
the document into windows makes every window compete to fill every declared field.
Measured on the same 572-char text, `purchase_date` came back as both `March 14, 2025`
and `last Saturday`, `pair_count` as both `3 pairs` and `Three pairs`, and
`personal_best` changed from `26:34` to `25:50` purely by changing window size from 220
to 400 chars. Nothing in the output says which is right.

**Q6 — 2.0.0 is the latest release and it already has "span attributes". No version
bump exists, let alone is needed.** `gliner2 2.0.0` is both installed and the newest on
PyPI (`pip index versions gliner2` → `LATEST: 2.0.0`). The feature upstream markets as
"span attributes" is `Schema.entity_attributes` — its own docstring says the labels are
"decoded as span attributes" (`inference/schema.py:331`), `AttributeGroup` is a public
lazy export (`__init__.py:42`), and it runs on our checkpoint today. The "2.5" in the
marketing is the **checkpoint generation** (`fastino/gliner2.5-base-v1`), not a library
version — and we already run that checkpoint. The checkpoint carries the weights all
three mechanisms need: `enable_records=True`, `enable_relations=True`,
`enable_count_head=True`, `enable_abstention=True` in its `BoundaryHeadSettings`. The
transformers floor in `gliner2_backend.py`'s docstring is untouched by any of this —
nothing here changes the dependency situation.

### What a decision-maker can rely on

- The joint path will never carry values. If #361 wants attributes, it is buying a
  second inference call per window, at roughly +90% wall clock, and reintroducing
  span-keyed matching between the two calls' outputs.
- `structure().field(dtype="str")` genuinely extracts the values #350 could not express
  — `25:50`, `3`, `8am-4pm`, `$129.99`, `March 14, 2025`, all five, first try, with
  offsets and calibrated-looking confidences.
- `entity_attributes` is **not** value extraction. It is per-span classification over a
  vocabulary you write in advance. It cannot produce `25:50` unless you already knew to
  write `25:50` into the schema.
- **There is a third option neither #350 nor this ticket assumed, and it works.**
  Declaring value-shaped *entity types* (`Duration`, `Quantity`, `TimeWindow`, `Money`,
  `Date`) and typed relations into them expresses all five facts on the **joint path**,
  one pass, with entity ids and #345's constraint enforcement intact. #350's "no
  expressible answer" is a property of #350's vocabulary, not of the API. It is not free:
  it got 2 of 4 values wrong on this text (see Q5/A below), and it cost 280ms vs 124ms.

### What is unverified

Everything here is mechanism, measured on **one** 572-char window. Comparative accuracy
of the three mechanisms is a sample of one and should not be read as a quality result.
The full list of unknowns is at the bottom.

---

## Evidence

### Q1 — No attribute mechanism exists on the joint path

`EntitySpec` carries six fields, none of them a value slot; `JointEntity` and
`JointRelation` likewise (`q1_joint_surface.py`):

```
EntitySpec             ['name', 'description', 'threshold', 'candidate_threshold', 'max_candidates', 'allow_nested']
RelationSpec           ['name', 'head', 'tail', 'description', 'threshold', 'candidate_threshold', 'directed', 'symmetric', 'inverse', 'allow_self', 'max_per_head', 'max_per_tail']
CompiledJointSchema    ['model_schema', 'entity_specs', 'relation_specs', 'constraints', 'entity_order', 'relation_order']
JointEntity            ['id', 'type', 'text', 'start', 'end', 'confidence', 'sentence_id', 'rescued']
JointRelation          ['type', 'head', 'tail', 'confidence', 'derived']
JointResult            ['text', 'entities', 'relations', 'default_include_confidence', 'default_include_spans', 'feasible']
```

`entity()` is strictly keyword-only with a fixed set (`joint_ie/schema.py:68-70`) —
unlike `relation()`, which does take `**aliases` (`joint_ie/schema.py:78-82`), so the
absence is deliberate, not an oversight:

```
entity       (self, name, description=None, *, threshold=None, candidate_threshold=None, max_candidates=None, allow_nested=None)
relation     (self, name, head, tail, description=None, *, threshold=None, ..., **aliases)
```

Every route in:

```
  FAIL  entity(..., attributes=...)            TypeError: JointSchema.entity() got an unexpected keyword argument 'attributes'
  FAIL  entity(..., fields=...)                TypeError: JointSchema.entity() got an unexpected keyword argument 'fields'
  FAIL  entity(..., choices=...)               TypeError: JointSchema.entity() got an unexpected keyword argument 'choices'
  FAIL  entity(..., dtype='str')               TypeError: JointSchema.entity() got an unexpected keyword argument 'dtype'
  FAIL  entity(..., validators=...)            TypeError: JointSchema.entity() got an unexpected keyword argument 'validators'
  FAIL  relation(..., attributes=...)          TypeError: unknown relation options: ['attributes']
  FAIL  schema.field('pb')                     AttributeError: 'JointSchema' object has no attribute 'field'
  FAIL  schema.structure('run')                AttributeError: 'JointSchema' object has no attribute 'structure'
  FAIL  schema.classification('t', ['a'])      AttributeError: 'JointSchema' object has no attribute 'classification'
  FAIL  schema.entity_attributes({...})        AttributeError: 'JointSchema' object has no attribute 'entity_attributes'
  FAIL  from_dict entity with 'attributes'     TypeError: JointSchema.entity() got an unexpected keyword argument 'attributes'
  OK    from_dict top-level 'attributes'
```

The last row is a silent-drop hazard worth a note: `JointSchema.from_dict` reads only
`entities`, `relations` and `constraints` (`joint_ie/schema.py:118-132`) and ignores any
other top-level key without complaint. A config file carrying an `attributes:` block
would load cleanly and do nothing.

What the compiler hands the model confirms it (`joint_ie/compiler.py:26-28`):

```
model_schema keys : ['classifications', 'entities', 'entity_descriptions', 'json_descriptions', 'json_structures', 'relations']
json_structures   : []
classifications   : []
```

Both are hard-coded empty — literally `"json_structures":[],"classifications":[]` in
the source. And forcing them full does not help. Appending a structure and a
classification to `compiled.model_schema` by hand and calling
`JointIEEngine.extract()` on it (`q245_joint_compose.py`):

```
model_schema now: {"json_structures": [{"facts": {"personal_best": "", "shift_window": ""}}], "classifications": [{"task": "topic", ...
extract() OK. to_dict keys: ['entities', 'relations']
entities: 14 relations: 5
anything named 'facts'/'topic' in the result?  False
```

It runs, and the result has nowhere to put either. `JointResult.to_dict()` returns
exactly two keys and the public surface has no value-bearing member:

```
JointResult.to_dict keys : ['entities', 'relations']
entity row keys          : ['confidence', 'end', 'id', 'start', 'text', 'type']
public JointResult attrs : ['default_include_confidence', 'default_include_spans', 'entities',
                            'entities_by_type', 'entity', 'feasible', 'get_entity', 'incoming',
                            'neighbors', 'outgoing', 'relations', 'relations_by_type',
                            'relations_of', 'text', 'to_dict', 'to_networkx']
```

### Q2 — The legacy API: three mechanisms, one pass, one trade

#### (a) `entity_attributes` — closed label sets on entity spans

`AttributeGroup(labels, multi_label=False, threshold=0.5, applies_to=None,
qualify_labels=False)` (`inference/schema.py:59-76`), attached via
`Schema.entity_attributes({name: AttributeGroup(...)})` (`inference/schema.py:325-407`).
Live (`q23_legacy.py`, probe A), `include_spans=True, include_confidence=True`:

```json
"Product": [
  {
    "text": "Saucony Endorphin Speed",
    "confidence": 0.5889124274253845,
    "start": 192,
    "end": 215,
    "sentiment": {"label": "positive", "confidence": 0.8831151127815247},
    "tags": [{"label": "footwear", "confidence": 0.5248549580574036}]
  }
]
```

The attribute lands **inside the entity row**, keyed by group name. `sentiment` is
single-label (one dict), `tags` is `multi_label=True` (a list).

Mechanically, the labels are injected into the model-facing entity vocabulary
(`inference/schema.py:400-402`) and then excluded from the *public* entity order at
decode (`models/boundary/engine.py:246-254`), so they never appear as entity types:

```
model-facing entity keys (attribute labels injected):
  ['Person', 'Organization', 'Location', 'Product', 'Event',
   'athlete', 'clinic', 'employee', 'footwear', 'negative', 'positive', 'race', 'vendor']
```

Then each attribute label is **force-scored at every retained entity span**
(`models/boundary/engine.py:379-485`, docstring at 388: *"Force-score configured
attribute labels at retained entity spans."*). Two validation rules worth knowing
before designing around it: labels must be globally unique across groups
(`inference/schema.py:376-380`), reproduced —

```
ValueError: Label 'unknown' is in both 'personal_best' and 'shift'
```

— and attribute labels must not collide with entity labels unless `qualify_labels=True`
(`inference/schema.py:388-393`).

#### (b) `structure().field()` — the only mechanism that extracts actual values

`StructureBuilder.field(name, dtype="list"|"str", choices=None, description=None,
threshold=None, validators=None, cardinality=None, exclusive=False)`
(`inference/schema.py:103-135`). Live, on the test text (`q23_legacy.py`, probe B),
**all six declared fields filled correctly on the first attempt**:

```json
{"run": [{
  "personal_best": {"text": "25:50",          "confidence": 0.9999144, "start": 82,  "end": 87},
  "previous_best": {"text": "26:34",          "confidence": 0.9997168, "start": 114, "end": 119},
  "shoe_count":    {"text": "3 pairs",        "confidence": 0.8612927, "start": 235, "end": 242},
  "price":         {"text": "$129.99",        "confidence": 0.9976240, "start": 313, "end": 320},
  "purchase_date": {"text": "March 14, 2025", "confidence": 0.9998848, "start": 324, "end": 338},
  "shift_window":  {"text": "8am-4pm",        "confidence": 0.9994596, "start": 564, "end": 571}
}]}
```

These are precisely the three fact shapes #350 reported as having no expressible
answer, plus a price and a date.

#### (c) `classification` — document-level, entity-free

```json
{"topic": {"label": "running", "confidence": 0.999951958656311}}
```

#### Same-pass composition, and what it costs

All of it in one `extract()` call (`q23_legacy.py`, probe D):

```
result keys: ['facts', 'entities', 'topic', 'relation_extraction']
```

with `facts.personal_best = "25:50"`, entity rows carrying `role: {"label": "clinician",
"confidence": 0.99999}`, `topic`, and `relation_extraction` all present together.

But this is the legacy runtime, and its relations are the untyped ones #345 retired.
From the built schema in the *same* call (`q25_tradeoff.py`):

```
built relations entry: [{"works_shift": {"head": "", "tail": ""}}, {"personal_best": {"head": "", "tail": ""}}]
relation_extraction:
    works_shift     head={"text": "Admon", "start": 540, "end": 545, "confidence": 0.96009} tail={"text": "Sunday shift", "start": 548, "end": 560, "confidence": 0.96009}
    personal_best   head={"text": "user", "start": 218, "end": 222, "confidence": 0.93307} tail={"text": "26:34", "start": 114, "end": 119, "confidence": 0.93307}
facts: [{"shift_window": {"text": "8am-4pm", ...}, "best_time": {"text": "25:50", ...}}]
```

Three things in that one block. Head/tail are span dicts, not ids — the re-matching
#345 removed. Head/tail typing is gone (`{"head": "", "tail": ""}`). And, in the same
pass on the same text, the *relation* mechanism answered `personal_best = 26:34` (the
old PB) and `works_shift = "Sunday shift"`, while the *structure* mechanism answered
`25:50` and `8am-4pm`. On this text the value fields were right and the relations were
wrong.

So: **same-pass values require giving up `JointSchema`.** Keeping `JointSchema` means
two calls — quantified under Q5.

### Q3 — Where a value lands, in each mechanism

| | `entity_attributes` | `structure().field()` | `classification` |
| --- | --- | --- | --- |
| value is | a **label** from a closed set | a **real text span** | a **label** from a closed set |
| offsets | **no** | yes (`start`/`end`) — except `choices=` fields | no |
| own confidence | yes | yes | yes |
| attached to | the entity row, in the legacy output | nothing — document-scoped | nothing — document-scoped |
| entity id | legacy output has no entity ids | — | — |
| cardinality | one label, or a list if `multi_label` | `dtype="str"` → one, `"list"` → all | one, or a list if `multi_label` |

An attribute value is exactly two keys (`q34_constraints.py`, probe 4b):

```
keys on an attributed entity: ['confidence', 'end', 'mood', 'planet', 'start', 'text']
keys inside an attribute value: ['confidence', 'label']
```

`models/boundary/engine.py:469-485` is the whole of it — a `{"label", "confidence"}`
dict for single-label, a filtered list for multi-label. No span is ever recorded for
the value itself, only for the entity it hangs on.

Structure field values come from `_format_structure_field`
(`models/boundary/engine.py:762-792`): scalar (`dtype="str"`) returns `spans[0]` with
offsets; list returns all of them; empty returns `None`.

**Re-matching.** The legacy output has no entity ids anywhere, so attaching a legacy
value to a `JointResult` entity is span-keyed matching — the apparatus #345 retired.
It does work, because both runtimes report the same character offsets
(`q34_constraints.py`, probe 4f):

```
joint entities (id, type, span):
    e1 Person       'user'                     [0,4)
    ...
record instances (anchor span -> value):
    anchor={'text': 'Admon', 'start': 414, 'end': 419} value=None
    anchor={'text': 'Admon', 'start': 540, 'end': 545} value={'text': '8am-4pm', 'start': 564, 'end': 571}
    span (414, 419) -> joint entity id 'e10'
    span (540, 545) -> joint entity id 'e14'
```

Exact `(start, end)` keys matched both anchors to joint entity ids. So the coupling is
mechanically recoverable — it is just not given.

**Record mode is the closest thing to a per-entity value**
(`inference/schema.py:246-265`, `processing/records.py:33-34`:
`VALID_MODES = ("natural", "latent", "anchorless")`). It produces one instance per
anchor mention (`q245_joint_compose.py`, probe 2):

```
  mode='natural'    -> {"shift_record": [{"person": {"text": "Admon", "start": 414, "end": 419}, "window": {"text": "8am-4pm", "start": 564, "end": 571}},
                                         {"person": {"text": "Admon", "start": 540, "end": 545}, "window": {"text": "8am-4pm", "start": 564, "end": 571}}]}
  mode='anchorless' -> {"shift_record": [{"person": {"text": "Admon", "start": 414, "end": 419}, "window": {"text": "8am-4pm", "start": 564, "end": 571}}]}
```

That is the right answer. But binding is not reliable: the same construct with a
`best_time` field bound the *runner's* PB to Admon, the physiotherapist, in both
instances, and left `shift` null (`q245_joint_compose.py`, probe 2b):

```json
{"person_facts": [
  {"name": {"text": "Admon", "start": 414, "end": 419}, "best_time": {"text": "25:50", "confidence": 0.889, "start": 82, "end": 87}, "shift": null},
  {"name": {"text": "Admon", "start": 540, "end": 545}, "best_time": {"text": "25:50", "confidence": 0.676, "start": 82, "end": 87}, "shift": null}
]}
```

Both `name` anchors are Admon; neither is the runner; the PB is 460 characters away.
Field-to-anchor assignment is a learned sparse-edge scoring step
(`processing/records.py:11-12`) with no locality or type constraint available to us.

### Q4 — How values are constrained, and when

**Closed label sets are enforced by construction, not by a decoder constraint.** An
`AttributeGroup`'s labels become the only rows that are scored
(`models/boundary/engine.py:398-406`), and a `choices=` field is scored only at the
positions where its declared choices appear in the schema prefix
(`models/boundary/engine.py:656-670`, docstring: *"Score enum values at their schema-prefix
positions."*). There is no analogue of `allow_edge` because there is no search to prune.

**A single-label attribute group can never abstain.** `models/boundary/engine.py:479-485`
takes `softmax(...).argmax()` — one of the declared labels always wins. With three
labels no entity in the text could plausibly carry (`q34_constraints.py`, probe 4a):

```
  Person   'Admon'                    planet={"label": "Mars", "confidence": 0.4931333661079407} mood=[]
  Person   'Admon'                    planet={"label": "Mars", "confidence": 0.481342613697052} mood=[]
  Product  'Saucony Endorphin'        planet={"label": "Mars", "confidence": 0.44202378392219543} mood=[]
```

Every entity is on Mars. The `mood` group — same probe, `multi_label=True,
threshold=0.9` — returned `[]` for all three, which is the *only* abstention route in
the entire surface (`models/boundary/engine.py:469-478`).

The same failure mode showed up earlier with value-shaped labels: enumerating
`["25:50", "26:34", "no time"]` and `["8am-4pm", "9am-5pm", "no shift"]` assigned
`shift: 8am-4pm` (conf 0.507) to Admon and `shift: 8am-4pm` (conf 0.614) to *"Saucony
Endorphin"*, a running shoe.

**`choices=` hallucinates.** A field whose choice list contains nothing present in the
text still returns a choice, at high confidence (`q34_constraints.py`, probe 4c):

```json
{"f": [{
  "shift_enum":  {"text": "8am-4pm", "confidence": 0.9999529123306274},
  "shift_free":  {"text": "8am-4pm", "confidence": 0.985742449760437, "start": 564, "end": 571},
  "absent_enum": {"text": "alpha",   "confidence": 0.9906994700431824}
}]}
```

`alpha`/`beta`/`gamma` appear nowhere in the text; `alpha` came back at 0.99. Note also
that choice-valued fields lose `start`/`end` — they are scored at *schema-prefix*
positions, not text positions, so there is no text span to report. Free-text fields
keep their offsets.

**`validators=` is a post-hoc filter, demonstrably.** Three runs of the same field
(`q34_constraints.py`, probe 4d):

```
  validator=matches 25:50      -> {"f": [{"t": {"text": "25:50", "confidence": 0.9882827401161194, "start": 82, "end": 87}}]}
  validator=matches nothing    -> {}
  validator=no validator       -> {"f": [{"t": {"text": "25:50", "confidence": 0.9882827401161194, "start": 82, "end": 87}}]}
```

Byte-identical confidence with and without the matching validator — the decode was not
steered, only screened afterwards. Source agrees: the validator runs inside the
span-collection loop after `token_boundaries_to_character_offsets`
(`models/boundary/engine.py:561-580`), and for entities at
`models/boundary/engine.py:292-296`.

**`cardinality=` and `exclusive=` are inert outside record mode.**
`inference/schema.py:116-119` states it: *"`cardinality` ... and `exclusive` refine
record decoding when the structure declares a record `mode`; they are ignored
otherwise."*

**There is no feasibility signal** (`q34_constraints.py`, probe 4e):

```
  all-absent structure -> {}
  partly-absent structure -> {"mixed": [{"blood_type": null, "shift_window": {"text": "8am-4pm", ...}}]}
  legacy result type: dict - has .feasible? False
```

An unfillable field is `null`. A structure with *every* field unfillable disappears
entirely — `models/boundary/engine.py:590-591`:

```python
if any(value is not None and value != [] for value in instance.values()):
    results[structure_name] = [instance]
```

No key, no log line, no counter. #350 established that prohibitive relation constraints
can never make a window infeasible; attributes do not change that, because the legacy
runtime has no `feasible` concept at all — it returns a plain `dict`, not a
`JointResult`.

### Q5 — Composition with our pipeline

**Per-window driving works.** We own the windowing, we call `extract()` per window, and
offsets come back window-local needing the `+offset` remap we already do
(`q56_windows_alt.py`, probe 5c, 4 windows of 220 chars, overlap 40):

```
  window@   0 -> {"personal_best": ["26:34", 114, 114], "purchase_date": ["last Saturday", 43, 43]}
  window@ 180 -> {"pair_count": ["3 pairs", 55, 235], "price": ["$129.99", 133, 313], "purchase_date": ["March 14, 2025", 144, 324]}
  window@ 360 -> {"shift_window": ["8am-4pm", 204, 564]}
  window@ 540 -> {"shift_window": ["8am-4pm", 24, 564]}
```

**But a structure is document-scoped, so windowing changes the answer.** Same text,
same schema, three drive strategies (`q25_tradeoff.py`, probe 5e):

```
  whole document:
    personal_best   '25:50'
    pair_count      '3'
    shift_window    '8am-4pm'
    price           '$129.99'
    purchase_date   'March 14, 2025'
  windows size=220 overlap=40:   (4 windows)
    pair_count      1 hits, distinct=['3 pairs']
    personal_best   1 hits, distinct=['26:34']
    price           1 hits, distinct=['$129.99']
    purchase_date   2 hits, distinct=['March 14, 2025', 'last Saturday']  <-- CONFLICT
    shift_window    2 hits, distinct=['8am-4pm']
  windows size=400 overlap=60:   (2 windows)
    pair_count      2 hits, distinct=['3 pairs', 'Three pairs']  <-- CONFLICT
    personal_best   1 hits, distinct=['25:50']
    ...
```

`personal_best` is `25:50` whole-document, `26:34` at window size 220, `25:50` again at
size 400. Two fields conflict across windows. Every window is asked to fill every
declared field and has no way to say "not here" — a window containing only
`"assistant: Three pairs is a reasonable rotation for that mileage."` still returns a
`facts` instance, with `pair_count: "Three pairs"` at 0.989 and four `null`s
(`q25_tradeoff.py`, probe 5f). Whatever consumes this needs its own cross-window
reconciliation for values, and the per-field confidence is the only signal it gets.

**Cost.** Median of 5 reps after a warmup, 572-char text, CPU (`q245_joint_compose.py`,
probe 3):

```
  JOINT entities+relations (JointIE)                   median=  157.3ms
  legacy entities only                                 median=  102.0ms
  legacy entities+relations                            median=  122.8ms
  legacy entities+relations+1 attr group (3 labels)    median=  134.5ms
  legacy entities+relations+structure(6 str fields)    median=  134.4ms
  legacy entities+relations+structure(18 str fields)   median=  211.2ms
  JOINT pass THEN legacy value pass (two calls)        median=  298.8ms
```

**Candidate expansion is linear, not quadratic.** Unlike permissive relation endpoints
(#350 measured that expansion), declaring attributes adds one model-facing query per
label and structure fields add one per field (`q245_joint_compose.py`, probe 3b):

```
  entities only                entity-queries=  5 relations=0 structure-fields=0
  entities+relations           entity-queries=  5 relations=2 structure-fields=0
  +1 attr group (3 labels)     entity-queries=  8 relations=2 structure-fields=0
  +structure 6 fields          entity-queries=  5 relations=2 structure-fields=6
  +structure 18 fields         entity-queries=  5 relations=2 structure-fields=18
```

6 fields cost +12ms, 18 fields +88ms — superlinear in wall clock beyond ~12 queries
but nowhere near the quadratic blow-up of all-types endpoints. The dominant cost is the
extra forward pass, not the extra queries.

### Q5/A — The alternative #350 did not try: value-shaped entity types

This is the finding most relevant to #361's "does this displace reified assertions"
question. Declaring `Duration`, `Quantity`, `TimeWindow`, `Money`, `Date` as **entity
types** and relating into them expresses all five facts on the **joint** path, one pass,
with ids and #345's constraint enforcement intact (`q56_windows_alt.py`, probe 5a):

```
  feasible=True entities=18 relations=13
    ENT e7   Quantity    '3 pairs'                  [235,242) conf=0.999
    ENT e8   Money       '$129.99'                  [313,320) conf=0.999
    ENT e9   Date        'March 14, 2025'           [324,338) conf=1.000
    ENT e14  TimeWindow  '8am-4pm'                  [458,465) conf=0.998
    ENT e3   Duration    '26:34'                    [114,119) conf=0.358
    REL owns_count     Person:'user' -> Quantity:'3 pairs' conf=0.985
    REL paid           Person:'user' -> Money:'$129.99' conf=0.964
    REL personal_best  Person:'user' -> Duration:'26:34' conf=0.971
    REL works_shift    Person:'Admon' -> TimeWindow:'Sunday shift' conf=0.879
```

All four question types now have an expressible answer, which #350's relation-only
vocabulary did not. Three caveats, all visible in that output:

1. **Two of four values are wrong.** `personal_best` bound `26:34` (the *previous* PB) —
   `25:50` was never extracted as a `Duration` at all. `works_shift` bound
   `'Sunday shift'` (conf 0.198–0.265 as an entity) rather than `'8am-4pm'`, which *was*
   extracted as a `TimeWindow` at 0.998 and sat two tokens away. The structure path got
   both right on the same text.
2. **Triplication.** Each relation appears 3–4 times at slightly different confidences
   (mention-level duplication, the same 3.3x effect #350 measured at corpus scale), and
   18 entities came out of a 572-char window.
3. **Cost.** 280ms vs 124ms for a 2-type/1-relation joint baseline
   (`q56_windows_alt.py`, probe 5d) — more than the 143ms legacy structure pass, though
   still less than the 299ms two-call split.

### Q6 — Version reality

```
  gliner2.__version__      : 2.0.0
  AttributeGroup exported? : True
  model.architecture       : boundary
  config.name_or_path      : fastino/gliner2.5-base-v1
```

```
$ pip index versions gliner2
gliner2 (2.0.0)
Available versions: 2.0.0, 1.3.2, 1.3.1, 1.3.0, 1.2.6, ...
  INSTALLED: 2.0.0
  LATEST:    2.0.0
```

2.0.0 is both what we run and the newest release. There is no version to bump to.

The "span attributes" upstream markets is `Schema.entity_attributes`, by the package's
own words (`inference/schema.py:328-332`):

> Attach attribute groups to entities declared by this schema.
>
> Model-facing attribute labels are added to the internal entity schema, but are
> excluded from the public entity order and **decoded as span attributes**.

`AttributeGroup` is a public lazy export (`__init__.py:42`), it is reachable from a
plain `import gliner2`, and it runs on our checkpoint — proved above. The "2.5" is the
**checkpoint generation** (`fastino/gliner2.5-base-v1`), not a library version, and we
already run that checkpoint. Its `BoundaryHeadSettings` carries every head the three
mechanisms need:

```
enable_records=True, record_dim=128, record_instance_queries=32,
record_anchor_threshold=0.5, record_field_threshold=0.5,
enable_relations=True, enable_count_head=True, enable_abstention=True,
abstention_threshold=0.5, classification_temperature=1.0, pair_temperature=1.0
```

**Nothing here touches the transformers floor.** The constraint recorded in
`unstructured2graph/src/unstructured2graph/gliner2_backend.py`'s module docstring —
`gliner2[local]` pins `transformers<5`, the workspace floors at `>=5.0.0rc3` for
CVE-2026-1839, so `gliner2` is a manual install and never a declared extra — is
unchanged by anything in this note. `Requires-Dist: transformers<5,>=4.38; extra ==
"local"` in the installed `METADATA` confirms the pin is still there in 2.0.0. Adopting
attributes requires no new dependency, no new extra and no new version.

---

## Summary table

| # | Question | Answer |
| --- | --- | --- |
| 1 | Attribute slot on the joint path? | **No.** No field on `EntitySpec`/`JointSchema`/`CompiledJointSchema`/`JointResult`; `entity()` has no `**aliases`; the compiler hard-codes `json_structures=[]`/`classifications=[]`; smuggled structures are silently dropped. |
| 2 | The legacy API | **Three mechanisms**: `entity_attributes` (closed labels on spans), `structure().field()` (real value spans), `classification` (doc-level labels). All compose with entities+relations in **one** call — but that call is the legacy runtime, whose relations are untyped span-dict pairs. Values in one pass **or** typed relations, not both. |
| 3 | Output shape | Attribute = `{"label","confidence"}`, **no offsets**, co-located with the entity row but no entity id. Structure field = real span with offsets — **document-scoped**, attached to nothing. Record mode gives per-anchor instances whose spans do match joint entity ids by `(start,end)`. |
| 4 | Constrained like endpoints? | **No `allow_edge` analogue, no `feasible` analogue.** Closed sets enforced by construction; regex types are a **post-hoc filter** (byte-identical confidence proves it); cardinality is `dtype`. Single-label groups **cannot abstain**; `choices=` **hallucinates** (0.99 on a label absent from the text). Unfillable → `null`; all-unfillable → structure silently dropped. |
| 5 | Composition | Per-window driving works; cost is +10% for a small attribute group or a 6-field structure, +72% for 18 fields, **+90% for the two-call split**. Candidate growth is **linear**, not quadratic. But structures are document-scoped: windowing produced two cross-window value conflicts and one window-size-dependent answer, with no signal. |
| 6 | Version reality | **2.0.0 is installed and is the latest on PyPI.** "Span attributes" = `Schema.entity_attributes`, present in 2.0.0, working on our checkpoint. "2.5" is the checkpoint generation, which we already run. No version bump; no change to the transformers floor. |

## Things I could not determine

- **Accuracy.** Every comparison here is one 572-char window. The structure path beat
  the relation path 5/5 vs 2/4 on this text; that is a sample of one and must not be
  read as a quality result. A real read needs #332's extraction-quality eval over the
  session corpus, which #350 already has the harness for.
- **Whether record mode's field-to-anchor binding can be fixed from the schema.** It
  bound a value 460 characters away to the wrong anchor. `occurrence_policy`
  (`"all" | "first" | "error_on_ambiguous" | "latent_all"`, `processing/records.py:34`)
  and `cardinality`/`exclusive` exist and were not swept — `error_on_ambiguous` in
  particular sounds like it might surface the ambiguity rather than guessing, but it was
  not exercised. This is the single most consequential unknown for #361, because record
  mode is the only construct that produces a per-entity value.
- **Whether attributes survive `extract_long`.** `inference/chunking.py:321-324` says
  attribute payloads are preserved verbatim through the long-text merge, but this was
  not run — we drive windows ourselves (#350/#352), so it only matters if that ever
  changes.
- **The span (non-boundary) architecture.** Attributes have a second, separate
  implementation there (`inference/runtime.py:563-675`), including a
  `_dedupe_attributed_entities` overlap pass (`runtime.py:704-716`) with no boundary
  equivalent. Not verified on a span checkpoint; only matters if we switch checkpoints.
- **Whether a value-shaped-entity-type vocabulary can be derived automatically.** The
  Q5/A result depends on hand-writing `Duration`/`Quantity`/`TimeWindow` with good
  descriptions. Whether #353's `LlmRecommendationStrategy` would ever propose
  value-shaped entity types, or how #359's range-width question interacts with them, is
  untouched here.
- **Calibration.** Structure-field confidences cluster at 0.99+ and attribute
  confidences much lower (0.44–0.99), but the two come from different heads
  (`pair_temperature` vs the entity head) and were not calibrated against each other.
  Thresholding them with a single number would be unsound.

---

## Appendix: the test text and the fixture

```python
# fixture.py
CKPT = "fastino/gliner2.5-base-v1"

TEXT = "\n\n".join([
    "user: I finally broke 26 minutes on the 5K last Saturday. My new "
    "personal best is 25:50, which beats my old PB of 26:34.",
    "assistant: That is a strong improvement. Are you still running in the "
    "Saucony Endorphin Speed?",
    "user: Yes. I own 3 pairs of them now. I bought the latest pair at "
    "Runner's World in Zagreb for $129.99 on March 14, 2025.",
    "assistant: Three pairs is a reasonable rotation for that mileage.",
    "user: Admon at the clinic works the Sunday shift, 8am-4pm, so I can "
    "only get a physio slot before 8am.",
    "assistant: I will note that Admon's Sunday shift is 8am-4pm.",
])   # len(TEXT) == 572

ENTITY_TYPES = ["Person", "Organization", "Location", "Product", "Event"]
```

The three mechanisms, minimally:

```python
from gliner2 import AutoExtractor
from gliner2.inference.schema import AttributeGroup, RegexValidator, Schema

model = AutoExtractor.from_pretrained(CKPT)          # NOT JointIE

# (a) closed labels bound to entity spans
s = Schema().entities(ENTITY_TYPES).entity_attributes({
    "role": AttributeGroup(labels=["athlete", "clinician"], applies_to=["Person"]),
})

# (b) real value spans -- document-scoped
s = Schema()
s.structure("facts").field("personal_best", dtype="str",
                           description="the runner's best 5K time")

# (c) per-anchor instances -- the only entity-attached value construct
s = Schema()
b = s.structure("person_facts", mode="natural", anchor="name")
b.field("name", dtype="str", cardinality="required_one")
b.field("shift", dtype="str", cardinality="optional_one")

out = model.extract(TEXT, s, include_spans=True, include_confidence=True)
```

Note `include_spans` / `include_confidence` are **kwargs** on the legacy
`extract()` (`inference/runtime.py:1159`), unlike the joint path where they live on
`JointIEConfig` — the reverse of the trap #345 recorded for `extract_long`.
