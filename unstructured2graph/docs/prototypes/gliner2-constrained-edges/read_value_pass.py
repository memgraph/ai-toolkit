"""#386: does a separate value-only pass keep the core value types' spans?

read_386.log showed the derived vocabularies losing value spans to domain
types: in S2V `25:50` is typed Food, in S1V it is not extracted at all. Labels
compete for a span inside one GLiNER2 pass, so this adds a second pass per
derived vocabulary holding only User, Person, the core value types, the heads
of that vocabulary's own relations into values, and those relations -- nothing
hand-written is added, the vocabulary stays derived.

Merge: a value span from the value pass overrides the main pass's typing of
any span it overlaps; main-pass edges with an endpoint on such a span drop
unless they already typed it as that value; the value pass's edges are added.

Same 10 evidence sessions, one turn per window, schemas built once and held
(#365). The hand arm must reproduce its 895 raw edges.

    python read_value_pass.py S1V S2V
"""

import json
import sys

import derive
import value_types as vt
from read_derived import agreement, report_arm

VALUES = set(vt.VALUE_ENTITIES)


def value_vocabulary(vocab):
    """The value-only slice of a derived vocabulary: its relations into values and every type they touch."""
    relations = [r for r in vocab["relations"] if set(r["tail"]) <= VALUES]
    labels = {"User", "Person", *VALUES} | {label for r in relations for label in r["head"]}
    described = {label: spec["description"] for label, spec in vocab["entity_types"].items()}
    return {label: described.get(label) or derive.CORE_ENTITIES[label] for label in labels}, relations


def span(sid, start, text):
    return sid, start, start + len(text)


def overlaps(a, b):
    return a[0] == b[0] and a[1] < b[2] and b[1] < a[2]


def merge(main, value):
    """Main-pass edges and mentions with the value pass's value spans taking precedence."""
    (main_edges, main_mentions), (value_edges, value_mentions) = main, value
    owned = {}
    for m in value_mentions:
        if m["type"] in VALUES:
            owned[span(m["sid"], m["start"], m["text"])] = m["type"]
    by_sid = {}
    for s, label in owned.items():
        by_sid.setdefault(s[0], []).append((s, label))

    def claim(sid, start, text):
        mine = span(sid, start, text)
        return next((label for s, label in by_sid.get(sid, ()) if overlaps(mine, s)), None)

    def keeps(e):
        for side in ("head", "tail"):
            label = claim(e["sid"], e[f"{side}_start"], e[f"{side}_text"])
            if label is not None and e[f"{side}_type"] != label:
                return False
        return True

    edges = [e for e in main_edges if keeps(e)]
    dropped = len(main_edges) - len(edges)
    seen = {(e["sid"], e["relation"], e["head_start"], e["tail_start"]) for e in edges}
    added = [e for e in value_edges if (e["sid"], e["relation"], e["head_start"], e["tail_start"]) not in seen]
    mentions = [m for m in main_mentions if claim(m["sid"], m["start"], m["text"]) in (None, m["type"])]
    retyped = len(main_mentions) - len(mentions)
    mentions += [m for m in value_mentions if m["type"] in VALUES]
    print(
        f"   merge: {len(owned)} value spans from the value pass; {retyped} main mentions overridden, "
        f"{dropped} main edges dropped, {len(added)} value edges added"
    )
    return edges + added, mentions


def main():
    names = tuple(sys.argv[1:]) or ("S1V", "S2V")
    vocabularies = {}
    for name in names:
        state = json.loads((derive.OUT / f"{name}.json").read_text())
        assert state.get("vocabulary"), f"derivation {name} did not pass validation"
        vocabularies[name] = state["vocabulary"]

    sample = json.loads((derive.HERE / "sample_sessions.json").read_text())
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
        value_types, value_relations = value_vocabulary(vocab)
        print(f"--- {name} value pass: {sorted(value_types)}")
        for r in value_relations:
            print(f"   {r['name']:<22}{r['head']} -> {r['tail']}")
        schemas[f"{name}+values"] = derive.build_schema(engine, value_types, value_relations)

    results = {name: derive.extract(engine, config, schema, docs) for name, schema in schemas.items()}
    assert len(results["hand"][0]) == 895, "hand arm must reproduce windowing.py's one-turn-each row"

    report_arm("hand", *results["hand"], meta)
    for name in names:
        print(f"\n##### {name}")
        report_arm(f"{name} alone", *results[name], meta)
        merged = merge(results[name], results[f"{name}+values"])
        report_arm(f"{name} + value pass", *merged, meta)
        (derive.OUT / f"eval_{name}_valuepass.json").write_text(
            json.dumps({"edges": merged[0], "mentions": merged[1]}, indent=1)
        )
    if len(names) == 2:
        agreement(results[names[0]][1], results[names[1]][1], *names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
