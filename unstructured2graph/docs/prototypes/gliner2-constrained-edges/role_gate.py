"""#358: which `User` mentions are the user?

Re-reads the constrained arm of `constrained_edges_output.json` (spans are in
document coordinates) and maps every `User` endpoint back to the turn it came
from, by replaying sessions-graph's combined_text construction against the
corpus's own per-turn roles. No model run.

Ground truth used for scoring: a mention is the user iff it is first person in a
user turn, or `you` in an assistant turn (the assistant addressing the user).
"""

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = Path.home() / ".cache/context-graph-eval/longmemeval-s-98d7416c24c7.json"
FIRST_PERSON = {"i", "me", "my", "myself", "mine", "user", "i'm", "i've"}


def norm(t):
    return " ".join(t.strip().lower().split())


def stream_records(path):
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        assert handle.read(1).strip() == "["
        buffer = ""
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return
            buffer += chunk
            while True:
                stripped = buffer.lstrip()
                if stripped[:1] in (",", ""):
                    buffer = stripped[1:]
                    continue
                if stripped[:1] == "]":
                    return
                try:
                    record, end = decoder.raw_decode(stripped)
                except ValueError:
                    buffer = stripped
                    break
                buffer = stripped[end:]
                yield record


def document(turns):
    """combined_text, plus each surviving turn's (start, end, role)."""
    unique = {}
    for turn in turns:
        text = f"{turn['role']}: {turn['content']}".strip()
        if text:
            unique.setdefault(hashlib.sha256(text.encode()).hexdigest(), (text, turn["role"]))
    spans, cursor = [], 0
    for text, role in unique.values():
        spans.append((cursor, cursor + len(text), role))
        cursor += len(text) + 2
    return "\n\n".join(t for t, _ in unique.values()), spans


def surface_class(s):
    if s in FIRST_PERSON:
        return "first-person"
    if s == "assistant":
        return "literal 'assistant'"
    if s == "you":
        return "'you'"
    return "third party"


def is_user(s, role):
    return (s in FIRST_PERSON and role == "user") or (s == "you" and role == "assistant")


def main():
    rows = json.loads((HERE / "constrained_edges_output.json").read_text())
    qids = {r["question_id"] for r in rows}
    sids = {r["session_id"] for r in rows}
    spans, texts = {}, {}
    for record in stream_records(CORPUS):
        if record["question_id"] not in qids:
            continue
        for sid, turns in zip(record["haystack_session_ids"], record["haystack_sessions"], strict=True):
            if sid in sids:
                texts[sid], spans[sid] = document(turns)
        if len(spans) == len(sids):
            break
    sample = {
        s["session_id"]: s["text"]
        for q in json.loads((HERE / "sample_sessions.json").read_text())
        for s in q["sessions"]
    }
    assert all(texts[s] == sample[s] for s in texts)
    print(f"document reconstruction verified byte-exact for {len(texts)} sessions\n")

    def role(sid, start):
        return next((r for lo, hi, r in spans[sid] if lo <= start < hi), "?")

    cell = Counter()
    also = defaultdict(Counter)
    for row in rows:
        for e in row["arms"]["constrained"]["edges"]:
            for side in ("head", "tail"):
                s, t = norm(e[f"{side}_text"]), e[f"{side}_type"]
                also[s][t] += 1
                if t == "User":
                    cell[(s, role(row["session_id"], e[f"{side}_span"][0]))] += 1
    total = sum(cell.values())

    agg = Counter()
    for (s, r), n in cell.items():
        agg[(surface_class(s), r)] += n
    print(f"constrained arm: {total} User endpoints, by surface class x speaking role\n")
    print(f"  {'class':<22}{'user turn':>11}{'assistant turn':>16}{'total':>8}")
    for c in ("first-person", "literal 'assistant'", "'you'", "third party"):
        u, a = agg[(c, "user")], agg[(c, "assistant")]
        print(f"  {c:<22}{u:>11}{a:>16}{u + a:>8}")

    print("\n  per surface form:")
    per = defaultdict(Counter)
    for (s, r), n in cell.items():
        per[s][r] += n
    for s, rs in sorted(per.items(), key=lambda kv: -sum(kv[1].values())):
        others = {t: c for t, c in also[s].items() if t != "User"}
        print(
            f"    {s!r:<16} user={rs['user']:<4} assistant={rs['assistant']:<4}"
            + (f" also typed {others}" if others else "")
        )

    print(f"\n  {'gate':<46}{'collapsed':>10}{'not-the-user admitted':>23}{'real-user missed':>18}")
    for name, pred in (
        ("#347 as decided: collapse all", lambda s, r: True),
        ("surface form only", lambda s, r: s in FIRST_PERSON),
        ("role only", lambda s, r: r == "user"),
        ("surface AND role", lambda s, r: s in FIRST_PERSON and r == "user"),
        ("surface AND role, plus 'you'@assistant", is_user),
    ):
        kept = sum(n for (s, r), n in cell.items() if pred(s, r))
        wrong = sum(n for (s, r), n in cell.items() if pred(s, r) and not is_user(s, r))
        missed = sum(n for (s, r), n in cell.items() if not pred(s, r) and is_user(s, r))
        print(f"  {name:<46}{kept:>10}{wrong:>23}{missed:>18}")

    print("\n  first person inside assistant turns (distinct spans, with context):")
    seen = set()
    for row in rows:
        sid = row["session_id"]
        for e in row["arms"]["constrained"]["edges"]:
            for side in ("head", "tail"):
                s, st = norm(e[f"{side}_text"]), e[f"{side}_span"][0]
                if (
                    e[f"{side}_type"] == "User"
                    and s in FIRST_PERSON
                    and role(sid, st) == "assistant"
                    and (sid, st) not in seen
                ):
                    seen.add((sid, st))
                    ctx = texts[sid][max(0, st - 90) : st + 60].replace("\n", " | ")
                    print(f"    ...{ctx}...\n        edge: {e['relation']}({e['head_text']!r} -> {e['tail_text']!r})")


if __name__ == "__main__":
    main()
