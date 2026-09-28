"""#359: what does a relation's declared range suppress, and can a rule tell the
real losses from the junk?

Reads the permissive arm of `constrained_edges_output.json` and tabulates every
edge the constrained arm's `start_labels`/`end_labels` would reject, grouped by
(relation, head type, tail type). Each group carries a verdict from reading its
examples; the two tests then ask whether a mechanical rule reproduces that
verdict. No model run.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import constrained_edges as ce

HERE = Path(__file__).parent
REL = {r.label: r for r in ce.RELATION_TYPES}

# Verdicts per violating (relation, head type, tail type) group, from its examples.
REAL = {  # true facts a wider range would recover
    ("visited", "User", "Organization"),  # museums: MoMA, Met, Brooklyn Museum
    ("located_in", "Product", "Location"),  # shoes -> closet, receipts -> home
    ("located_in", "Person", "Location"),  # frida kahlo -> mexico
    ("prefers", "User", "Organization"),  # dropbox, google drive, levi's
    ("attended", "Person", "Organization"),  # frida kahlo -> academy of san carlos
    ("attended", "User", "Location"),  # i -> gym
}
LABEL_CONFUSION = {  # a correct relation IS declared with these types (#360's territory)
    ("visited", "User", "Event"),  # attended covers it
    ("attended", "User", "Organization"),  # business administration: studied covers it
    ("visited", "User", "Person"),
    ("visited", "Person", "Person"),
    ("studied", "User", "Product"),
}


def norm(t):
    return " ".join(t.strip().lower().split())


def verdict(key):
    return "real" if key in REAL else "label-confusion" if key in LABEL_CONFUSION else "junk"


def main():
    rows = json.loads((HERE / "constrained_edges_output.json").read_text())
    raw = {arm: sum(len(r["arms"][arm]["edges"]) for r in rows) for arm in ("constrained", "permissive")}
    assert raw == {"constrained": 1004, "permissive": 567}, f"not the clean run (#365): {raw}"

    types_of = defaultdict(Counter)
    for row in rows:
        for arm in ("constrained", "permissive"):
            for e in row["arms"][arm]["edges"]:
                types_of[norm(e["head_text"])][e["head_type"]] += 1
                types_of[norm(e["tail_text"])][e["tail_type"]] += 1

    per_relation, groups, examples = Counter(), Counter(), defaultdict(set)
    ok = Counter()
    ambiguity = Counter()
    for row in rows:
        for e in row["arms"]["permissive"]["edges"]:
            r = REL[e["relation"]]
            per_relation[e["relation"]] += 1
            bad = [(s, a) for s, a in (("head", r.start_labels), ("tail", r.end_labels)) if e[f"{s}_type"] not in a]
            if not bad:
                ok[e["relation"]] += 1
                continue
            key = (e["relation"], e["head_type"], e["tail_type"])
            groups[key] += 1
            examples[key].add(f"{norm(e['head_text'])!r}->{norm(e['tail_text'])!r}")
            also_declared = all(any(types_of[norm(e[f"{s}_text"])][t] for t in allowed) for s, allowed in bad)
            ambiguity[(verdict(key), also_declared)] += 1

    print("permissive edges the declared ranges suppress, per relation (* = undeclared endpoint type)\n")
    for label, r in REL.items():
        v = sum(c for k, c in groups.items() if k[0] == label)
        print(
            f"{label:<11} ({','.join(r.start_labels)}) -> ({','.join(r.end_labels)})  conformant={ok[label]} suppressed={v}"
        )
        for key, c in sorted(((k, c) for k, c in groups.items() if k[0] == label), key=lambda kc: -kc[1]):
            h = key[1] + ("*" if key[1] not in r.start_labels else "")
            t = key[2] + ("*" if key[2] not in r.end_labels else "")
            print(f"    {c:>3}  {verdict(key):<16}{h:>14} -> {t:<14} {sorted(examples[key])[:3]}")

    totals = Counter()
    for key, c in groups.items():
        totals[verdict(key)] += c
    print(f"\nsuppressed: {sum(totals.values())}  " + "  ".join(f"{v}={n}" for v, n in totals.most_common()))
    org = sum(c for k, c in groups.items() if verdict(k) == "real" and "Organization" in k[1:])
    print(f"real losses with an Organization endpoint: {org}/{totals['real']}")

    print("\nTEST 1 -- frequency: does a group's share of its relation separate real from junk?")
    for key, c in sorted(groups.items(), key=lambda kc: -kc[1] / per_relation[kc[0][0]]):
        if c >= 5:
            print(f"    {c / per_relation[key[0]]:>5.0%}  {verdict(key):<16} {key[0]}({key[1]} -> {key[2]}) x{c}")

    print("\nTEST 2 -- type ambiguity: is the undeclared endpoint's text typed with a declared type anywhere?")
    for v in ("real", "label-confusion", "junk"):
        yes, no = ambiguity[(v, True)], ambiguity[(v, False)]
        print(f"    {v:<16} yes={yes:<4} no={no:<4} ({yes / max(yes + no, 1):.0%})")

    print("\nthe model's typing is consistent, not noisy -- every type a real-loss endpoint receives:")
    for text in ("brooklyn museum", "metropolitan museum of art", "museum of modern art", "levi's", "dropbox"):
        print(f"    {text!r:<32} {dict(types_of[text])}")


if __name__ == "__main__":
    main()
