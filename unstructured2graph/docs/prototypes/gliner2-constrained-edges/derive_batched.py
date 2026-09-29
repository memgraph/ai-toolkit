"""#372: run #353's derivation contract as revised by #366.

    propose      (LLM x K)  K disjoint time-stratified batches of whole sessions
    consolidate  (LLM)      merge the K proposals, synonyms merged; the only
                            stage allowed to rename
    observe      (GLiNER2)  one permissive pass, one turn per window (#352),
                            both candidate caps at 4096 (#371)
    prune        (LLM)      unchanged from derive.py
    validate                unchanged, plus a reporting-only line per core
                            value type with no relation into it

`python derive_batched.py S1` and `... S2` run two full derivations whose
propose batches are disjoint (the stability check); both observe derive.py's
60-session sample, so observe tables are comparable with #366's A and B. Stages
are cached in derivation/<name>.json. read_derived.py reads the result:
`python read_derived.py S1 S2`.
"""

import json
import random
import sys
import time

import derive
import value_types as vt
from windowing import turns_of

SEED = 372
K = 4
DERIVATIONS = ("S1", "S2")

VALUE_RULES = """Value facts: when the corpus states a fact about the user as a value (a time, a duration, an amount, a count, a price, a date, a time window: "my best 5K is 25:50", "I spent $400 on it"), propose a relation into the matching core value type (Duration, Quantity, Money, Date, TimeWindow). Every value fact the corpus states needs one; a value type nothing points into is never extracted.

No modal or tense variants: one relation per kind of fact, whatever its tense or modality. Do not propose plans_to_X, wants_to_X, will_X, used_to_X or considering_X beside X; the fact's time is recorded separately, and a variant fires on the same pairs as its base and contradicts it."""

PROPOSE_PROMPT = derive.PROPOSE_PROMPT.replace("\n\nConversations:", f"\n\n{VALUE_RULES}\n\nConversations:")
assert PROPOSE_PROMPT != derive.PROPOSE_PROMPT

CONSOLIDATE_PROMPT = """{k} independent proposals for one memory-graph vocabulary follow, each made from a different sample of the same corpus of conversations between a user and an AI assistant. Merge them into one vocabulary.

A fixed core is always present; do not redefine or rename it:
{core}

- Keep every distinct concept any proposal has: a domain one sample lacked is still in the corpus.
- Merge synonyms into one label or name (e.g. JobRole and Occupation, Food and Dish, visited and went_to). Pick the clearest name; renaming is allowed here and nowhere later.
- Relations: the extractor sees ONLY THE NAME, so the name alone must say what the relation means and which way it points. Union the intended_head/intended_tail of merged relations. Relations are binary and directed; no symmetric or inverse relations.
- Prefer the fewest types and relations that keep every concept; every extra label costs the extractor recall on the others.

{rules}

Proposals:
{proposals}"""


def batches():
    """2*K disjoint propose batches, each one session per time stratum under derive's prompt budget.

    The observe sample and the evidence sessions are excluded. Batches 0..K-1 are S1's, K..2K-1 S2's.
    """
    observe, _ = derive.samples()
    excluded = derive.excluded_sessions() | set(observe)
    pool = {}
    for record in vt.stream_records(vt.CORPUS):
        for sid, date, turns in zip(
            record["haystack_session_ids"], record["haystack_dates"], record["haystack_sessions"], strict=True
        ):
            if sid not in excluded and sid not in pool:
                pool[sid] = (date, sum(len(t["content"]) for t in turns))
    ordered = sorted((date, sid, chars) for sid, (date, chars) in pool.items())
    rng = random.Random(SEED)
    rows = [[] for _ in range(2 * K)]
    for stratum in derive.strata(ordered, derive.PROPOSE_STRATA):
        for batch, row in zip(rows, rng.sample(stratum, 2 * K), strict=True):
            batch.append(row)
    for batch in rows:  # whole sessions only: drop the longest until under budget
        while sum(chars for _, _, chars in batch) > derive.PROPOSE_CHARS:
            batch.remove(max(batch, key=lambda row: row[2]))
    return observe, [[sid for _, sid, _ in batch] for batch in rows]


def unanchored_value_types(vocabulary):
    """Core value types no kept relation points into: reporting-only (#366)."""
    tails = {label for r in vocabulary["relations"] for label in r["tail"]}
    return [label for label in vt.VALUE_ENTITIES if label not in tails]


def main(name):
    derive.OUT.mkdir(exist_ok=True)
    path = derive.OUT / f"{name}.json"
    state = json.loads(path.read_text()) if path.exists() else {}

    def save():
        path.write_text(json.dumps(state, indent=1))

    core = "\n".join(f"- {label}: {description}" for label, description in derive.CORE_ENTITIES.items())
    if "samples" not in state:
        observe, all_batches = batches()
        mine = all_batches[:K] if name == DERIVATIONS[0] else all_batches[K:]
        state["samples"] = {"observe": observe, "propose_batches": mine}
        save()
    print(f"[{name}] propose {[len(b) for b in state['samples']['propose_batches']]} sessions per batch", flush=True)

    state.setdefault("proposals", [])
    for i, sids in enumerate(state["samples"]["propose_batches"][len(state["proposals"]) :], len(state["proposals"])):
        docs = derive.load_texts(sids)
        sessions = "\n\n".join(
            f"=== conversation on {date} ===\n{turns_of(turns)[0]}"
            for date, turns in sorted(docs.values(), key=lambda d: d[0])
        )
        vocab, provenance = derive.ask(
            PROPOSE_PROMPT.format(n=len(docs), core=core, sessions=sessions), derive.PROPOSE_SCHEMA
        )
        state["proposals"].append({"vocabulary": vocab, "provenance": provenance})
        save()
        print(
            f"[{name}] batch {i}: {len(vocab['entity_types'])} types, {len(vocab['relations'])} relations", flush=True
        )

    if "propose" not in state:
        proposals = json.dumps([p["vocabulary"] for p in state["proposals"]], indent=1)
        vocab, provenance = derive.ask(
            CONSOLIDATE_PROMPT.format(k=K, core=core, rules=VALUE_RULES, proposals=proposals), derive.PROPOSE_SCHEMA
        )
        state["propose"] = {"vocabulary": vocab, "provenance": provenance}
        save()
    vocab = state["propose"]["vocabulary"]
    print(
        f"[{name}] consolidated {len(vocab['entity_types'])} domain types, {len(vocab['value_types'])} value types, "
        f"{len(vocab['relations'])} relations",
        flush=True,
    )

    if "observe" not in state:
        from gliner2.joint_ie import JointIE, JointIEConfig

        engine = JointIE.from_pretrained(vt.MODEL)
        config = JointIEConfig(
            include_spans=True,
            include_confidence=True,
            relation_pair_cap=derive.PAIR_CAP,
            max_edges_per_type=derive.PAIR_CAP,
        )
        everything = tuple(derive.entity_types(vocab))
        schema = derive.build_schema(  # held for the whole run (#365)
            engine,
            derive.entity_types(vocab),
            [{"name": r["name"], "head": everything, "tail": everything} for r in vocab["relations"]],
        )
        docs = derive.load_texts(state["samples"]["observe"])
        started = time.perf_counter()
        edges, mentions = derive.extract(engine, config, schema, docs)
        state["observe"] = {
            "tables": derive.observation_tables(vocab, edges, mentions),
            "seconds": round(time.perf_counter() - started),
            "edges": len(edges),
            "mentions": len(mentions),
            "chars": sum(len(turns_of(t)[0]) for _, t in docs.values()),
            "pair_cap": derive.PAIR_CAP,
        }
        save()
    observed = state["observe"]
    print(f"[{name}] observe {observed['edges']} edges, {observed['mentions']} mentions in {observed['seconds']}s")

    if "prune" not in state:
        prompt = derive.PRUNE_PROMPT.format(
            vocab=json.dumps(vocab, indent=1),
            n=len(state["samples"]["observe"]),
            core_identity=json.dumps(derive.CORE_IDENTITY),
            tables=json.dumps(observed["tables"], indent=1),
        )
        pruned, provenance = derive.ask(prompt, derive.PRUNE_SCHEMA)
        state["prune"] = {"result": pruned, "provenance": provenance}
        save()
    pruned = state["prune"]["result"]

    errors = derive.validate(vocab, pruned, observed["tables"])
    state["validation"] = errors
    state["vocabulary"] = None if errors else derive.final_vocabulary(vocab, pruned)
    state["unanchored_value_types"] = unanchored_value_types(state["vocabulary"]) if state["vocabulary"] else None
    stages = [p["provenance"] for p in state["proposals"]] + [
        state["propose"]["provenance"],
        state["prune"]["provenance"],
    ]
    state["cost_usd"] = round(sum(p["cost_usd"] for p in stages), 3)
    save()
    print(f"[{name}] validation: {'PASS' if not errors else 'FAIL'}")
    for error in errors:
        print(f"   {error}")
    print(f"[{name}] core value types with no relation into them: {state['unanchored_value_types']}")
    print(f"[{name}] LLM cost ${state['cost_usd']}, observe {observed['seconds']}s")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
