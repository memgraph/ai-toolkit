"""#352: does the windowing regime change what extraction can see?

Runs #361's value vocabulary (value_types.ARMS["values"]) over the 10 evidence
sessions under several windowing regimes: word windows of 192/384/768 words,
turn-aligned windows (whole turns packed up to 384 words; only a turn longer
than that is split), and per-turn windows (every turn alone, so no window ever
mixes speakers). One schema, held for the whole run (#365). The 384/64 word
regime must reproduce value_types.py's values arm: 1121 raw edges (its "954" is
the original eleven relations only).

Beyond counts, it reports what #358's gate cannot see: an edge whose head is in
a user turn and whose tail is in an assistant turn, and of those, how many have
a tail the user never wrote anywhere in the session (ungrounded: the user
cannot have asserted it).
"""

import hashlib
import json
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import value_types as vt
from role_gate import stream_records

HERE = Path(__file__).parent
CORPUS = Path.home() / ".cache/context-graph-eval/longmemeval-s-98d7416c24c7.json"
ANSWERS = {  # (question type, a regex an answer-bearing edge's endpoints must match)
    "single-session-user": r"business administration",
    "knowledge-update": r"25:50",
    "single-session-assistant": r"admon.*8 am|8 am.*admon",
    "temporal-reasoning": r"museum of modern art|moma",
}


@dataclass(frozen=True)
class Window:
    text: str
    start_char: int


def turns_of(turns):
    """combined_text plus each surviving turn's (start, end, role), as reconciliation builds it."""
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


def word_windows(engine, text, size, overlap):
    from gliner2.inference.chunking import split_text_into_chunks
    from gliner2.processing.word_splitter import word_splitter_from

    return [
        Window(w.text, w.start_char)
        for w in split_text_into_chunks(text, size, overlap, word_splitter=word_splitter_from(engine))
    ]


def turn_windows(engine, text, spans, size=384, overlap=64, pack=True):
    """Pack whole turns up to `size` words (or one turn per window when not
    `pack`); split only a turn that alone exceeds `size`."""
    out, group = [], []

    def words(span):  # span is (start, end, role)
        return len(text[span[0] : span[1]].split())

    def flush():
        if group:
            lo, hi = group[0][0], group[-1][1]
            out.append(Window(text[lo:hi], lo))
            group.clear()

    for span in spans:
        if words(span) > size:
            flush()
            for w in word_windows(engine, text[span[0] : span[1]], size, overlap):
                out.append(Window(w.text, span[0] + w.start_char))
            continue
        if group and (not pack or sum(map(words, group)) + words(span) > size):
            flush()
        group.append(span)
    flush()
    return out


def extract(engine, config, schema, windows):
    edges, mentions = [], 0
    for w in windows:
        joint = engine.extract(w.text, schema, config=config)
        by_id = {e.id: e for e in joint.entities}
        mentions += len(joint.entities)
        for r in joint.relations:
            h, t = by_id.get(r.head), by_id.get(r.tail)
            if h is not None and t is not None:
                edges.append((r.type, h.type, h.text, h.start + w.start_char, t.type, t.text, t.start + w.start_char))
    return edges, mentions


def main():
    from gliner2.joint_ie import JointIE, JointIEConfig

    sample = json.loads((HERE / "sample_sessions.json").read_text())
    meta = {s["session_id"]: q for q in sample for s in q["sessions"]}
    qids = {q["question_id"] for q in sample}
    docs = {}
    for record in stream_records(CORPUS):
        if record["question_id"] not in qids:
            continue
        for sid, turns in zip(record["haystack_session_ids"], record["haystack_sessions"], strict=True):
            if sid in meta:
                docs[sid] = turns_of(turns)
        if len(docs) == len(meta):
            break

    engine = JointIE.from_pretrained(vt.MODEL)
    config = JointIEConfig(include_spans=True, include_confidence=True)
    schema = vt.build_schema(engine, *vt.ARMS["values"])
    regimes = {
        "words 192/32": lambda text, spans: word_windows(engine, text, 192, 32),
        "words 384/64": lambda text, spans: word_windows(engine, text, 384, 64),
        "words 768/128": lambda text, spans: word_windows(engine, text, 768, 128),
        "turns <=384": lambda text, spans: turn_windows(engine, text, spans),
        "one turn each": lambda text, spans: turn_windows(engine, text, spans, pack=False),
    }
    print(
        f"{'regime':<15}{'windows':>8}{'mentions':>10}{'raw':>6}{'distinct':>10}{'cross-turn':>12}"
        f"{'user->asst':>12}{'ungrounded':>12}{'secs':>7}   answer-bearing"
    )
    results = {}
    for name, make in regimes.items():
        t0 = time.perf_counter()
        n_windows = mentions = 0
        edges_all, answers = [], Counter()
        for sid, (text, spans) in docs.items():
            windows = make(text, spans)
            n_windows += len(windows)
            edges, m = extract(engine, config, schema, windows)
            mentions += m
            user_text = " ".join(vt.norm(text[lo:hi]) for lo, hi, role in spans if role == "user")

            def turn_of(pos, spans=spans):
                return next(((i, r) for i, (lo, hi, r) in enumerate(spans) if lo <= pos < hi), (-1, "?"))

            for e in edges:
                (hi_, hr), (ti_, tr) = turn_of(e[3]), turn_of(e[6])
                u2a = hi_ != ti_ and hr == "user" and tr == "assistant"
                ungrounded = u2a and vt.norm(e[5]) not in user_text
                edges_all.append((sid, e, hi_ != ti_, u2a, ungrounded))
                blob = f"{vt.norm(e[2])} {vt.norm(e[5])}"
                pattern = ANSWERS.get(meta[sid]["question_type"])
                if pattern and re.search(pattern, blob):
                    answers[meta[sid]["question_type"]] += 1
        distinct = {(sid, e[0], e[1], vt.norm(e[2]), e[4], vt.norm(e[5])) for sid, e, *_ in edges_all}
        cross = sum(x[2] for x in edges_all)
        u2a = sum(x[3] for x in edges_all)
        ungrounded = sum(x[4] for x in edges_all)
        results[name] = len(edges_all)
        print(
            f"{name:<15}{n_windows:>8}{mentions:>10}{len(edges_all):>6}{len(distinct):>10}"
            f"{cross:>7} ({100 * cross / max(len(edges_all), 1):.0f}%){u2a:>12}{ungrounded:>12}"
            f"{time.perf_counter() - t0:>7.0f}   {dict(answers)}",
            flush=True,
        )
    assert results["words 384/64"] == 1121, "384/64 must reproduce value_types.py's values arm"
    del schema
    return 0


if __name__ == "__main__":
    sys.exit(main())
