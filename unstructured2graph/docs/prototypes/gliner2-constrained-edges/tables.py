"""#352: a markdown table spreads one fact across a column header, a row label
and a cell, which span-pair extraction cannot join. Does rewriting each row as
a sentence, for extraction only, let the fact bind? Runs #361's value vocabulary
over the Admon session's turns (one turn per window, as #352 decided), with and
without linearization. One schema held (#365).
"""

import json
import re
import sys
from pathlib import Path

import value_types as vt
import windowing as wd
from role_gate import stream_records

HERE = Path(__file__).parent
SESSION = "answer_sharegpt_5Lzox6N_0"
ROW = re.compile(r"^\s*\|(.*)\|\s*$")
RULE = re.compile(r"^\s*\|?\s*:?-{3,}")


TEMPLATES = {
    "generic": "{label}, {header}: {cell}.",
    "with a verb": "{cell} is assigned to the {header} on {label}.",
}


def linearize(text, template=TEMPLATES["generic"]):
    """Rewrite each markdown table body row as one sentence per cell, using `template`."""
    out, header = [], None
    for line in text.split("\n"):
        m = ROW.match(line)
        if not m:
            header = None
            out.append(line)
            continue
        if RULE.match(line):
            continue
        cells = [c.strip() for c in m.group(1).split("|")]
        if header is None:
            header = cells
            continue
        label, values = cells[0], cells[1:]
        out.append(
            " ".join(
                template.format(label=label, header=h, cell=v) for h, v in zip(header[1:], values, strict=False) if v
            )
        )
    return "\n".join(out)


def main():
    from gliner2.joint_ie import JointIE, JointIEConfig

    sample = json.loads((HERE / "sample_sessions.json").read_text())
    qid = next(q["question_id"] for q in sample for s in q["sessions"] if s["session_id"] == SESSION)
    turns = next(
        t
        for r in stream_records(wd.CORPUS)
        if r["question_id"] == qid
        for sid, t in zip(r["haystack_session_ids"], r["haystack_sessions"], strict=True)
        if sid == SESSION
    )
    engine = JointIE.from_pretrained(vt.MODEL)
    config = JointIEConfig(include_spans=True, include_confidence=True)
    schema = vt.build_schema(engine, *vt.ARMS["values"])
    arms = [("as written", lambda t: t)] + [
        (f"linearized, {name}", lambda t, tpl=tpl: linearize(t, tpl)) for name, tpl in TEMPLATES.items()
    ]
    for arm, fix in arms:
        rewritten = [dict(t, content=fix(t["content"])) for t in turns]
        text, spans = wd.turns_of(rewritten)
        edges, _ = wd.extract(engine, config, schema, wd.turn_windows(engine, text, spans, pack=False))
        shifts = sorted({f"{e[0]}({e[1]}:{e[2]!r} -> {e[4]}:{e[5]!r})" for e in edges if e[0] == "works_shift"})
        admon = sorted(
            {f"{e[0]}({e[1]}:{e[2]!r} -> {e[4]}:{e[5]!r})" for e in edges if "admon" in (vt.norm(e[2]) + vt.norm(e[5]))}
        )
        print(f"\n== {arm}: {len(edges)} edges; works_shift {len(shifts)}; edges touching Admon {len(admon)}")
        for s in (shifts + [a for a in admon if a not in shifts])[:24]:
            print(f"    {s}")
    for name, tpl in TEMPLATES.items():
        rows = [ln for ln in linearize("\n".join(t["content"] for t in turns), tpl).split("\n") if "Admon" in ln]
        print(f"\n{name} row: {rows[1][:160] if len(rows) > 1 else rows}")
    del schema
    return 0


if __name__ == "__main__":
    sys.exit(main())
