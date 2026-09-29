"""#366: why derivation's permissive observe pass found no user facts.

At gliner2's defaults, observe gave 245 edges over 709k chars, a tenth of
#350's permissive density. Heads of user-centric relations came out typed
Activity or Date, so prune (rightly) kept one relation of twelve. This runs
over the 10 evidence sessions, where the sample is known to be fact-dense, and
separates the suspects:

  1. vocabulary size x windowing, permissive: #350's 8 types vs #361's 13, on
     384/64 word windows and one turn per window. The 8-type word-window cell
     must reproduce #350's permissive arm (567 raw, 375 User heads).
  2. the candidate caps. gliner2 cuts relation candidates to relation_pair_cap
     (128) and max_edges_per_type (256), breaking score ties on
     str((entity_type, start, end)) (candidates.py:353). A permissive endpoint
     gives each span one equal-scoring copy per type, so past 11 types (121
     copies per span pair) the cut is alphabetical and User sorts last. The
     constrained 13-type row at the defaults must reproduce 895.
  3. derivation A's proposal, permissive (as observe ran it) and with its own
     intended endpoints.

Every schema is built once and held for the whole run (#365).
"""

import json
import sys
from collections import Counter

import derive
import value_types as vt
from windowing import turns_of, word_windows

BIG = 4096


def run(engine, config, schema, docs, words):
    """derive.extract, optionally on 384/64 word windows instead of one turn per window."""
    if not words:
        return derive.extract(engine, config, schema, docs)[0]
    edges = []
    for sid, (_, turns) in docs.items():
        text, _ = turns_of(turns)
        for w in word_windows(engine, text, 384, 64):
            joint = engine.extract(w.text, schema, config=config)
            by_id = {e.id: e for e in joint.entities}
            edges += [{"sid": sid, "head_type": by_id[r.head].type} for r in joint.relations if r.head in by_id]
    return edges


def main():
    sample = json.loads((derive.HERE / "sample_sessions.json").read_text())
    docs = derive.load_texts({s["session_id"] for q in sample for s in q["sessions"]})
    chars = sum(len(turns_of(t)[0]) for _, t in docs.values())
    vocab = json.loads((derive.OUT / "A.json").read_text())["propose"]["vocabulary"]

    from gliner2.joint_ie import JointIE, JointIEConfig

    engine = JointIE.from_pretrained(vt.MODEL)
    default = JointIEConfig(include_spans=True, include_confidence=True)
    raised = JointIEConfig(include_spans=True, include_confidence=True, relation_pair_cap=BIG, max_edges_per_type=BIG)

    def permissive(types, relations):
        every = tuple(types)
        return vt.build_schema(engine, types, [(r[0], every, every, "") for r in relations])

    values_types, values_relations = vt.ARMS["values"]
    a_types = derive.entity_types(vocab)
    a_every = tuple(a_types)
    schemas = {
        "8 permissive": permissive(vt.BASE_ENTITIES, vt.BASE_RELATIONS),
        "13 permissive": permissive(values_types, values_relations),
        "13 constrained": vt.build_schema(engine, values_types, values_relations),
        "A permissive": derive.build_schema(
            engine, a_types, [{"name": r["name"], "head": a_every, "tail": a_every} for r in vocab["relations"]]
        ),
        "A intended": derive.build_schema(
            engine,
            a_types,
            [{"name": r["name"], "head": r["intended_head"], "tail": r["intended_tail"]} for r in vocab["relations"]],
        ),
    }
    cells = [
        ("8 permissive", "words", default),
        ("8 permissive", "turn", default),
        ("13 permissive", "words", default),
        ("13 permissive", "turn", default),
        ("13 permissive", "turn", raised),
        ("13 constrained", "turn", default),
        ("13 constrained", "turn", raised),
        ("A permissive", "turn", default),
        ("A permissive", "turn", raised),
        ("A intended", "turn", default),
    ]
    print(f"10 evidence sessions, {chars} chars; caps default (128/256) or raised ({BIG})")
    print(f"{'schema':<16}{'windows':<8}{'caps':<9}{'raw':>6}{'User head':>11}   top head types")
    got = {}
    for name, windows, config in cells:
        edges = run(engine, config, schemas[name], docs, words=windows == "words")
        heads = Counter(e["head_type"] for e in edges)
        caps = "default" if config is default else "raised"
        got[(name, windows, caps)] = (len(edges), heads["User"])
        print(
            f"{name:<16}{windows:<8}{caps:<9}{len(edges):>6}{heads['User']:>6} ({100 * heads['User'] / max(len(edges), 1):>3.0f}%)"
            f"   {heads.most_common(4)}",
            flush=True,
        )
    assert got[("8 permissive", "words", "default")] == (567, 375), "must reproduce #350's permissive arm"
    assert got[("13 constrained", "turn", "default")][0] == 895, "must reproduce windowing.py's one-turn-each row"
    return 0


if __name__ == "__main__":
    sys.exit(main())
