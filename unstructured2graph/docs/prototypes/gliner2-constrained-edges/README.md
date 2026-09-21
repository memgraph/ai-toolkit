# #350 — edges to read

Throwaway prototype for [#350](https://github.com/memgraph/ai-toolkit/issues/350)
(map [#344](https://github.com/memgraph/ai-toolkit/issues/344)). Branch only, no PR —
same convention as the [#345 research note](https://github.com/memgraph/ai-toolkit/blob/research/gliner2-jointschema/unstructured2graph/docs/research/gliner2-jointschema.md).
Nothing here ships: it exists so the typed relation model could be judged against real
session text before the `hygm` plan is declared ready.

To re-run (needs `gliner2>=2.0.0` installed manually — see `gliner2_backend.py`'s module
docstring — and the cached `longmemeval-s` corpus under `~/.cache/context-graph-eval/`):

```bash
python sample_corpus.py          # writes sample_sessions.json (regenerable, not committed)
python constrained_edges.py      # writes full_run.log + constrained_edges_output.json (1.1MB, not committed)
python analyze.py                # writes analysis.log
python feasibility_probe.py      # the feasible=False reachability table in section D
```

10 evidence sessions (157K chars) from `longmemeval-s`, one per question type, run
twice through a typed `JointSchema`: **constrained** (real `start_labels`/`end_labels`)
and **permissive** (all 8 entity types on both endpoints — #345's only expressible
form of "unconstrained", and the plan's `start_labels=()` translation).

Full triples: `full_run.log`. Aggregates: `analysis.log`. Raw: `constrained_edges_output.json`.

| | constrained | permissive |
|---|---|---|
| raw edges | 1004 | 672 |
| distinct after merge | 304 | 328 |
| windows infeasible | 0/109 | 0/109 |
| wall clock | 75s | 78s |
| shared claims | 214 (14 of them endpoint-retyped) | |
| only in one arm | 85 constrained-only | 103 permissive-only |
| would fail #348's post-hoc check | — | 194 (29% of raw) |

---

## A. The edges that carry an answer

Q "What degree did I graduate with?" → *Business Administration*

    constrained   studied(User:'user' -> Organization:'Business Administration')   conf 0.99
    permissive    studied + attended + knows(User:'user' -> Organization:'Business Administration')

The answer lands. Note the endpoint is typed **Organization**, not `Topic` — a degree
name read as an institution. `studied`'s range happened to allow both.

Q "days between my MoMA visit and the Ancient Civilizations exhibit?"

    constrained   attended(User:'user' -> Event:'MoMA tour')
    permissive    attended(User:'user' -> Event:'MoMA tour')
                  visited(User:'user' -> Organization:'Museum of Modern Art')   <-- constrained arm lost this

`visited`'s declared range is `(Location,)`; the model types "Museum of Modern Art" as
`Organization`; the edge is therefore suppressed at decode time. **This is real,
answer-relevant signal lost to a domain/range binding that was one label too narrow.**

Q "personal best time in the charity 5K?" → *25:50* — **no edge in either arm.**
Q "how many items of clothing to pick up or return?" → *3* — **no edge in either arm.**
Q "what was Admon's Sunday rotation?" → *8am–4pm day shift* — **no edge in either arm.**

3 of 6 question types have no expressible answer in a relation vocabulary at all: the
answer is a *value* (a time, a count, a shift), not a link between two entities.

---

## B. What the constraint bought (85 constrained-only claims)

Suppressed junk that the permissive arm emitted, all flagged by the same binding
compiled post-hoc:

    works_for(Product:'Trello' -> Organization:'new job')
    works_for(Product:'slow cooker' -> Activity:'meal prep')
    works_for(Organization:'Mint' -> Organization:'Mint')
    knows(User:'user' -> Organization:'Business Administration')
    attended(User:'user' -> Organization:'Business Administration')
    lives_in(Product:'Lumetri Color Panel' -> Product:'Lumetri Color Panel')

Edges only the constrained arm found — #348's "the beam reallocates rather than
filters" claim, and they are not junk:

    purchased(User:'user' -> Product:"chef's knife")
    prefers(User:'me' -> Product:'Shoeboxed')
    practices(User:'me' -> Activity:'tracking expenses')
    owns(User:'user' -> Product:'produce')

And 14 claims survive both arms with **different endpoint typing**, which a text-keyed
diff would miss entirely:

    practices('user' -> 'routine'):          (User,Topic) permissive -> (User,Activity) constrained
    prefers('user' -> 'personal capital'):   (User,Organization)     -> (User,Product)

---

## C. Three problems in what came back

### C1. `User` is not the user

Every turn is prefixed `user: ` / `assistant: ` (#328), so the literal token is extracted
as a `User` entity and heads most edges. Over the 10 sessions, `User`-typed endpoints:

    502 first-person forms ('user', 'i', 'me')
     12 the literal 'assistant'
     85 third parties: 'kahlo' x38, 'frida' x33, 'frida kahlo' x11, 'you' x2, 'magdy' x1

So:

    owns(User:'assistant' -> Product:'Trello')          <-- the assistant recommended Trello
    visited(User:'kahlo' -> Location:'Paris')    x5     <-- Frida Kahlo visited Paris

#347 decided processing collapses **all** `User` mentions onto the session's `(:User)`
node. Applied to this output, that writes *"the user visited Paris"* and *"the user owns
Trello"* — false memories from an art-history digression and an assistant suggestion.
14% of `User` endpoints here are not the user.

### C2. Relation labels are not discriminated

14% of distinct pairs in the constrained arm (21% permissive) carry more than one label:

    'user' -> 'meal prep':    practices, prefers, studied
    'user' -> 'yoga pants':   owns, prefers, purchased
    'user' -> 'tennis racket':owns, prefers, purchased
    'frida' -> 'retablos':    owns, prefers, purchased, studied

The model asserts every plausible label rather than choosing. A derived vocabulary
(#347/#353) with near-synonymous relation types multiplies edges instead of sharpening
them, and `prefers` — the one the preference questions need — is indistinguishable from
`owns`.

### C3. Mention duplication and self-loops persist

3.3x duplication (1004 raw → 304 distinct). 2253 mentions → 2067 span-scoped nodes →
903 merged under #346's rule (2.5 mentions/node). Highest-degree merged nodes are
`User:user` (364), `User:i` (132), `Person:frida` (103) — hubs, as #346 predicted, and
the top two are the same entity fragmented by surface form ('user' vs 'i'), which
exact-normalized-text-within-type will never join.

12 constrained edges have endpoints with identical text:

    works_for(Person:'Zara' -> Organization:'Zara')
    visited(Person:'Agent' -> Location:'Agent')
    owns(Person:'Blazer' -> Product:'blazer')

`allow_self=False` (the default) rejects self-pairs by *entity id*, not by text, so two
mentions of one string still pair up.

---

## D. `feasible=False`: reachable, but not by this spec

The main run saw 0/109 infeasible windows in both arms. `feasible` is False only when
*every* candidate assignment fails `validate_solution` (`optimizers/beam.py:96-109`), and
the empty solution always validates — so prohibitive constraints can never force it.
Probe over 12 windows (`feasibility_probe.py`):

| schema | infeasible | relations kept |
|---|---|---|
| A: domain/range only | 0/12 | 64 |
| B: + `max_per_head=1`, `max_per_tail=1`, `no_self_loops`, `at_most(per_head=1)` | 0/12 | 20 |
| C: + `symmetric`/`inverse` companions against those caps | **6/12** | 4 |

- Under the plan's spec shape (domain/range, optionally cardinality) the infeasible-window
  retry #348 designed is **unreachable** — B drops 68% of edges silently and still
  reports `feasible=True`.
- Completeness constraints (symmetric/inverse) do reach it, and when they do it is
  catastrophic: half the windows return empty.
- Also learned: a `symmetric` relation is **rejected at schema build** unless its head and
  tail type sets are compatible (`ValueError: symmetric relations require compatible head
  and tail types`) — a symmetric relation type forces domain == range in the plan's model.
