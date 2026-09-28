"""#358 question 5: is the `user: `/`assistant: ` role prefix load-bearing?

Same 10 sessions, same constrained vocabulary as constrained_edges.py; the only
variable is whether each turn carries its role prefix. The prefixed arm must
reproduce constrained_edges.py's constrained arm exactly (1004 raw edges).
"""

import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import constrained_edges as ce
from role_gate import is_user, norm, stream_records, surface_class

HERE = Path(__file__).parent
CORPUS = Path.home() / ".cache/context-graph-eval/longmemeval-s-98d7416c24c7.json"


def build(turns, *, prefixed):
    """combined_text and per-turn (start, end, role). Dedupe hashes the unit
    actually sent, exactly as reconciliation does -- itself part of what the
    prefix changes."""
    unique = {}
    for turn in turns:
        body = turn["content"].strip()
        if not body:
            continue
        text = f"{turn['role']}: {body}" if prefixed else body
        unique.setdefault(hashlib.sha256(text.encode()).hexdigest(), (text, turn["role"]))
    spans, cursor = [], 0
    for text, role in unique.values():
        spans.append((cursor, cursor + len(text), role))
        cursor += len(text) + 2
    return "\n\n".join(t for t, _ in unique.values()), spans


def main():
    sample = json.loads((HERE / "sample_sessions.json").read_text())
    qids = {q["question_id"] for q in sample}
    meta = {s["session_id"]: q for q in sample for s in q["sessions"]}
    turns_by = {}
    for record in stream_records(CORPUS):
        if record["question_id"] not in qids:
            continue
        for sid, turns in zip(record["haystack_session_ids"], record["haystack_sessions"], strict=True):
            if sid in meta:
                turns_by[sid] = turns
        if len(turns_by) == len(meta):
            break

    from gliner2.joint_ie import JointIE, JointIEConfig

    engine = JointIE.from_pretrained(ce.MODEL)
    schema = ce.build_schema(engine, permissive=False)  # one schema for both arms, held (see ce.build_schema)
    rows = []
    for sid, turns in turns_by.items():
        row = {
            "session_id": sid,
            "question_type": meta[sid]["question_type"],
            "answer": meta[sid]["answer"],
            "arms": {},
        }
        for arm, prefixed in (("prefixed", True), ("bare", False)):
            text, spans = build(turns, prefixed=prefixed)
            t0 = time.perf_counter()
            session = {"session_id": sid, "text": text, "date": ""}
            result = ce.run_arm(engine, JointIEConfig, session, permissive=False, max_windows=None)
            row["arms"][arm] = {
                "spans": spans,
                "edges": [
                    {
                        "relation": e.relation,
                        "head_type": e.head_type,
                        "head_text": e.head_text,
                        "head_start": e.head_span[0],
                        "tail_type": e.tail_type,
                        "tail_text": e.tail_text,
                        "tail_start": e.tail_span[0],
                    }
                    for e in result.edges
                ],
            }
            print(
                f"  {sid:<28} {arm:<9} {len(text):>6}ch {len(result.edges):>5} edges {time.perf_counter() - t0:5.1f}s",
                flush=True,
            )
        rows.append(row)
    del schema
    report(rows)
    return 0


def report(rows):
    print()
    print(f"  {'':<44}{'prefixed':>10}{'bare':>10}")
    stats = {}
    for arm in ("prefixed", "bare"):
        raw = 0
        claims, claims_global = set(), set()
        cell = Counter()
        answer_hits = Counter()
        for row in rows:
            spans = row["arms"][arm]["spans"]
            edges = row["arms"][arm]["edges"]
            raw += len(edges)

            def role(start, spans=spans):
                return next((r for lo, hi, r in spans if lo <= start < hi), "?")

            tokens = {t for t in re.findall(r"[a-z0-9']+", str(row["answer"]).lower()) if len(t) > 3}
            for e in edges:
                claim = (e["relation"], e["head_type"], norm(e["head_text"]), e["tail_type"], norm(e["tail_text"]))
                claims.add((row["session_id"], *claim))
                claims_global.add(claim)
                for side in ("head", "tail"):
                    if e[f"{side}_type"] == "User":
                        cell[(norm(e[f"{side}_text"]), role(e[f"{side}_start"]))] += 1
                blob = f"{norm(e['head_text'])} {norm(e['tail_text'])}"
                if tokens and any(t in blob for t in tokens):
                    answer_hits[row["question_type"]] += 1
        endpoints = sum(cell.values())
        passed = sum(n for (s, r), n in cell.items() if is_user(s, r))
        third_in_user = sum(n for (s, r), n in cell.items() if surface_class(s) == "third party" and r == "user")
        stats[arm] = {
            "raw edges": raw,
            "distinct typed claims, all sessions": len(claims_global),
            "distinct typed claims, per session": len(claims),
            "User endpoints": endpoints,
            "pass the #358 gate": passed,
            "fail the gate": endpoints - passed,
            "third parties inside user turns": third_in_user,
            "'i' as User": sum(n for (s, r), n in cell.items() if s == "i"),
            "third parties typed User": sum(n for (s, r), n in cell.items() if surface_class(s) == "third party"),
            "answer-bearing (single-session-user)": answer_hits["single-session-user"],
            "answer-bearing (single-session-preference)": answer_hits["single-session-preference"],
        }
    for key in stats["prefixed"]:
        print(f"  {key:<44}{stats['prefixed'][key]:>10}{stats['bare'][key]:>10}")
    assert stats["prefixed"]["raw edges"] == 1004, "prefixed arm must reproduce the constrained arm"

    for arm in ("prefixed", "bare"):
        print(f"\n  {arm}: third parties inside user turns")
        for row in rows:
            spans = row["arms"][arm]["spans"]
            for e in row["arms"][arm]["edges"]:
                for side in ("head", "tail"):
                    s = norm(e[f"{side}_text"])
                    r = next((r for lo, hi, r in spans if lo <= e[f"{side}_start"] < hi), "?")
                    if e[f"{side}_type"] == "User" and surface_class(s) == "third party" and r == "user":
                        print(f"      {e['relation']}(User:{e['head_text']!r} -> {e['tail_type']}:{e['tail_text']!r})")


if __name__ == "__main__":
    sys.exit(main())
