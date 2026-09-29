"""#366: read the derived vocabularies against the hand-written one.

Runs three constrained vocabularies over the 10 evidence sessions, one turn
per window (#352): the hand-written stand-in (#361's value vocabulary, which
must reproduce windowing.py's "one turn each" row, 895 raw edges) and the two
derivations derive.py produced. None of the 10 sessions was in any derivation
sample. Every schema is built once and held for the whole run (#365).

Reports what #366 asks to read: counts, answer-bearing edges (and #359's MoMA
`visited` edge), each relation's pairs with examples, co-firing (#360), User
endpoints that are not first person (#358), the identity calls, and the two
derivations' behavioural agreement by span co-assignment (#353).
"""

import json
import re
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

import derive
import value_types as vt
from role_gate import FIRST_PERSON
from windowing import ANSWERS

HERE = Path(__file__).parent
DERIVATIONS = tuple(sys.argv[1:]) or ("A", "B")
TENSE = re.compile(r"^(plans?_to|wants?_to|will|used_to|considering|intends?_to|going_to)_(.+)$")


def distinct(edges):
    return {
        (e["sid"], e["relation"], e["head_type"], vt.norm(e["head_text"]), e["tail_type"], vt.norm(e["tail_text"]))
        for e in edges
    }


def show(e):
    return (
        f"{e['relation']}({e['head_type']}:{e['head_text']!r} -> {e['tail_type']}:{e['tail_text']!r}) {e['confidence']}"
    )


def report_arm(name, edges, mentions, meta):
    print(f"\n=== {name}: {len(edges)} raw, {len(distinct(edges))} distinct, {len(mentions)} mentions")
    answers = [
        e
        for e in edges
        if (pattern := ANSWERS.get(meta[e["sid"]]["question_type"]))
        and re.search(pattern, f"{vt.norm(e['head_text'])} {vt.norm(e['tail_text'])}")
    ]
    print(f"answer-bearing: {len(answers)}")
    for e in sorted({show(e) for e in answers}):
        print(f"   {e}")
    moma = sorted({show(e) for e in edges if re.search(r"museum of modern art|moma", vt.norm(e["tail_text"]))})
    print(f"edges into MoMA: {moma or 'none'}")

    by_relation = defaultdict(list)
    for e in edges:
        by_relation[e["relation"]].append(e)
    print("relations (raw, top pairs, examples):")
    for relation, mine in sorted(by_relation.items(), key=lambda kv: -len(kv[1])):
        pairs = Counter(f"{e['head_type']}->{e['tail_type']}" for e in mine).most_common(3)
        examples = list(dict.fromkeys(f"{e['head_text']} -> {e['tail_text']}" for e in mine))[:4]
        print(f"   {relation:<22}{len(mine):>4}  {pairs}  {examples}")

    labels = defaultdict(set)
    for e in edges:
        labels[(e["sid"], vt.norm(e["head_text"]), vt.norm(e["tail_text"]))].add(e["relation"])
    multi = [ls for ls in labels.values() if len(ls) > 1]
    co_fire = Counter(label for ls in multi for label in ls)
    fired = Counter(label for ls in labels.values() for label in ls)
    print(
        f"co-firing: {len(multi)}/{len(labels)} pairs multi-label; "
        f"top {[(label, f'{co_fire[label]}/{fired[label]}') for label, _ in co_fire.most_common(4)]}"
    )
    print(f"   combos {Counter(tuple(sorted(ls)) for ls in multi).most_common(4)}")

    user_ends = [
        (e[f"{side}_text"], e["sid"]) for e in edges for side in ("head", "tail") if e[f"{side}_type"] == "User"
    ]
    not_first = Counter(vt.norm(t) for t, _ in user_ends if vt.norm(t) not in FIRST_PERSON | {"you"})
    print(
        f"User endpoints not first person: {sum(not_first.values())}/{len(user_ends)} "
        f"({100 * sum(not_first.values()) / max(len(user_ends), 1):.1f}%)  {not_first.most_common(6)}"
    )


def tense_variants(relations):
    """Modal/tense relation names (plans_to_X...) and the base relation each shadows, if declared (#366)."""
    names = {r["name"] for r in relations}
    found = []
    for name in sorted(names):
        if m := TENSE.match(name):
            base = m.group(2)
            found.append((name, sorted(n for n in names if n != name and (n == base or base in n or n in base))))
    return found


def agreement(mentions_a, mentions_b, a="A", b="B"):
    """Align a's types to b's by the spans both typed, and report how much they agree."""
    typed_a = {(m["sid"], m["start"], vt.norm(m["text"])): m["type"] for m in mentions_a}
    typed_b = {(m["sid"], m["start"], vt.norm(m["text"])): m["type"] for m in mentions_b}
    shared = typed_a.keys() & typed_b.keys()
    matrix = Counter((typed_a[k], typed_b[k]) for k in shared)
    best = {}
    for (ta, tb), n in matrix.items():
        if n > best.get(ta, (None, 0))[1]:
            best[ta] = (tb, n)
    agreed = sum(n for _, n in best.values())
    print(
        f"\n=== stability: {len(shared)} spans typed by both, {len(typed_a.keys() - shared)} only by {a}, "
        f"{len(typed_b.keys() - shared)} only by {b}; agreement under best {a}->{b} type map {agreed}/{len(shared)} "
        f"({100 * agreed / max(len(shared), 1):.0f}%)"
    )
    totals = Counter(typed_a[k] for k in shared)
    for ta, (tb, n) in sorted(best.items(), key=lambda kv: -totals[kv[0]]):
        others = [(bb, nn) for (aa, bb), nn in matrix.most_common() if aa == ta and bb != tb][:3]
        print(f"   {a}:{ta:<18} -> {b}:{tb:<18} {n}/{totals[ta]}  also {others}")


def main():
    vocabularies = {}
    for name in DERIVATIONS:
        state = json.loads((derive.OUT / f"{name}.json").read_text())
        assert state.get("vocabulary"), f"derivation {name} did not pass validation"
        vocabularies[name] = state["vocabulary"]
        print(f"--- derivation {name}")
        for label, spec in state["vocabulary"]["entity_types"].items():
            stats = state["observe"]["tables"]["types"].get(label, {})
            print(
                f"   {label:<18}{spec['identity']:<8}mentions {stats.get('mentions', 0):>5}  "
                f"recurring {stats.get('texts_recurring_across_sessions')}  caps {stats.get('capitalized_share')}  "
                f"{spec['description'][:70]}"
            )
        for rel in state["vocabulary"]["relations"]:
            print(f"   {rel['name']:<22}{rel['head']} -> {rel['tail']}")
        dropped = state["prune"]["result"]
        print(f"   dropped relations {[r['name'] for r in dropped['dropped_relations']]}")
        print(f"   dropped types {[t['label'] for t in dropped['dropped_types']]}")
        print(f"   tense variants {tense_variants(state['vocabulary']['relations'])}")
        if "unanchored_value_types" in state:
            print(f"   core value types with no relation into them {state['unanchored_value_types']}")

    sample = json.loads((HERE / "sample_sessions.json").read_text())
    meta = {s["session_id"]: q for q in sample for s in q["sessions"]}
    docs = derive.load_texts(meta)

    from gliner2.joint_ie import JointIE, JointIEConfig

    engine = JointIE.from_pretrained(vt.MODEL)
    config = JointIEConfig(include_spans=True, include_confidence=True)
    hand_types, hand_relations = vt.ARMS["values"]
    schemas = {"hand": vt.build_schema(engine, hand_types, hand_relations)}
    for name, vocab in vocabularies.items():
        types = {label: spec["description"] for label, spec in vocab["entity_types"].items()}
        schemas[name] = derive.build_schema(engine, types, vocab["relations"])

    results = {name: derive.extract(engine, config, schema, docs) for name, schema in schemas.items()}
    assert len(results["hand"][0]) == 895, "hand arm must reproduce windowing.py's one-turn-each row"
    for name, (edges, mentions) in results.items():
        report_arm(name, edges, mentions, meta)
        (derive.OUT / f"eval_{name}.json").write_text(json.dumps({"edges": edges, "mentions": mentions}, indent=1))
    for a, b in pairwise(DERIVATIONS):
        agreement(results[a][1], results[b][1], a, b)
    return 0


if __name__ == "__main__":
    sys.exit(main())
