"""#360: what steers a relation's precision on the joint path?

gliner2's joint compiler writes relations as {name: {"head": "", "tail": ""}} and
passes only ENTITY descriptions (joint_ie/compiler.py:27-28), so a relation's
description should be inert and its name the only lever. Tests both on `prefers`,
everything else fixed. Every schema built is held for the whole run (#365).
"""

import dataclasses
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import constrained_edges as ce
from multilabel import PREFERS_TRUE

HERE = Path(__file__).parent
PREFERS_FALSE = {
    ("user", "tennis racket"),
    ("user", "yoga pants"),
    ("user", "tops"),
    ("user", "short-sleeve shirts"),
    ("i", "short-sleeve shirts"),
    ("user", "gold"),
    ("user", "mint"),
    ("user", "audio advanced settings"),
    ("user", "advanced settings"),
    ("user", "lumetri color panel"),
    ("user", "meal prep"),
}
SHARP = "explicitly says they like, love, enjoy or favour it -- not merely owning, buying, using or asking about it"
VARIANTS = (
    ("baseline", "prefers", None),
    ("sharp description", "prefers", SHARP),
    ("renamed", "likes", None),
    ("renamed", "says_they_like", None),
)


def norm(t):
    return " ".join(t.strip().lower().split())


def main():
    from gliner2.joint_ie import JointIE, JointIEConfig

    sample = json.loads((HERE / "sample_sessions.json").read_text())
    engine = JointIE.from_pretrained(ce.MODEL)
    original = ce.RELATION_TYPES
    held, results = [], {}
    for kind, name, description in VARIANTS:
        ce.RELATION_TYPES = tuple(
            dataclasses.replace(r, label=name, description=description or r.description) if r.label == "prefers" else r
            for r in original
        )
        ce._SCHEMAS.clear()
        held.append(ce.build_schema(engine, permissive=False))
        pairs, raw, edges = defaultdict(set), Counter(), []
        for q in sample:
            for s in q["sessions"]:
                for e in ce.run_arm(engine, JointIEConfig, s, permissive=False, max_windows=None).edges:
                    pairs[(norm(e.head_text), norm(e.tail_text))].add(e.relation)
                    raw[e.relation] += 1
                    edges.append(
                        (s["session_id"], e.relation, e.head_type, norm(e.head_text), e.tail_type, norm(e.tail_text))
                    )
        results[(kind, name)] = (pairs, raw, edges)
        target = {k for k, v in pairs.items() if name in v}
        others = sum(c for r, c in raw.items() if r != name)
        heads = Counter(k[0] for k in target)
        print(
            f"{kind:<18} {name:<15} raw={raw[name]:<4} pairs={len(target):<4} "
            f"true kept {len(target & set(PREFERS_TRUE))}/{len(PREFERS_TRUE)}  "
            f"false kept {len(target & PREFERS_FALSE)}/{len(PREFERS_FALSE)}  "
            f"other relations raw={others}  top heads={heads.most_common(3)}",
            flush=True,
        )

    base = results[("baseline", "prefers")]
    assert sum(base[1].values()) == 1004, "baseline must reproduce the constrained arm"
    identical = results[("sharp description", "prefers")][2] == base[2]
    print(
        f"\nsharp description vs baseline: edge lists identical = {identical} "
        "(expected: the compiler drops relation descriptions)"
    )
    del held
    return 0


if __name__ == "__main__":
    sys.exit(main())
