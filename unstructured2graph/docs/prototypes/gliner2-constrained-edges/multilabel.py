"""#360: when several relation labels fire on one (head, tail) pair, are the
extra labels noise?

Lists every multi-label pair in the constrained arm with per-label confidence,
then prints the source window behind each co-fired `prefers` beside a verdict
from reading it. No model run.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent

# From reading each window below.
PREFERS_TRUE = {
    ("user", "black jeans"): "my new black jeans from Levi's, which I'm really loving",
    ("i", "black jeans"): "same sentence",
    ("i", "meal prep"): "I'm especially interested in meal prep",
    ("user", "sarcophagi"): "I was particularly fascinated by the sarcophagi",
    ("frida", "retablos"): "Frida was particularly drawn to retablos and ex-votos",
    ("frida", "ex-votos"): "same sentence",
}


def norm(t):
    return " ".join(t.strip().lower().split())


def main():
    rows = json.loads((HERE / "constrained_edges_output.json").read_text())
    assert sum(len(r["arms"]["constrained"]["edges"]) for r in rows) == 1004, "not the clean run (#365)"
    text = {
        s["session_id"]: s["text"]
        for q in json.loads((HERE / "sample_sessions.json").read_text())
        for s in q["sessions"]
    }

    conf = defaultdict(lambda: defaultdict(float))
    where = {}
    for row in rows:
        for e in row["arms"]["constrained"]["edges"]:
            key = (norm(e["head_text"]), norm(e["tail_text"]))
            conf[key][e["relation"]] = max(conf[key][e["relation"]], e["confidence"] or 0.0)
            if e["relation"] == "prefers":
                where.setdefault(key, (row["session_id"], e["tail_span"]))
    multi = {k: v for k, v in conf.items() if len(v) > 1}
    print(f"{len(conf)} distinct pairs, {len(multi)} multi-label ({100 * len(multi) / len(conf):.0f}%)\n")
    for combo, c in Counter(tuple(sorted(v)) for v in multi.values()).most_common():
        print(f"    {c:>3}  {combo}")

    print("\nevery multi-label pair, labels by confidence:")
    for key, labels in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        ranked = ", ".join(f"{r} {c:.2f}" for r, c in sorted(labels.items(), key=lambda x: -x[1]))
        print(f"    {key[0]!r:>14} -> {key[1]!r:<30} {ranked}")

    prefers_pairs = {k for k, v in conf.items() if "prefers" in v}
    cofired = sorted(k for k in prefers_pairs if len(conf[k]) > 1)
    print(
        f"\nprefers: {len(prefers_pairs)} pairs, co-fires with another label on {len(cofired)} "
        f"({100 * len(cofired) / len(prefers_pairs):.0f}%) -- the reporting-only co-firing signal"
    )
    print(f"co-fired prefers judged true: {sum(k in PREFERS_TRUE for k in cofired)}/{len(cofired)}\n")
    for key in cofired:
        sid, (start, end) = where[key]
        window = text[sid][max(0, start - 150) : end + 50].replace("\n", " | ")
        mark = "TRUE " if key in PREFERS_TRUE else "false"
        print(f"  [{mark}] {key[0]!r} -> {key[1]!r}  prefers {conf[key]['prefers']:.2f}, labels {sorted(conf[key])}")
        print(f"          ...{window[-210:]}...")


if __name__ == "__main__":
    main()
