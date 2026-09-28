"""#361: do value-shaped entity types actually recall the values?

#362 declared `Duration`/`Quantity`/`Money` as entity types on the joint path and
got 2 of 4 values bound wrong -- in a sample of one 572-char window. This runs the
same idea over #350's 10 real evidence sessions, against #350's own vocabulary as
the baseline, and reads:

  1. do the answers #350 called unexpressible now land
  2. does adding 5 types + 8 relations degrade the original 11 relations
  3. what the value types extract
  4. how often value nodes would collide within a session (decision 2's premise)
"""

import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
CORPUS = Path.home() / ".cache/context-graph-eval/longmemeval-s-98d7416c24c7.json"
OUT = HERE / "value_types.json"
MODEL = "fastino/gliner2.5-base-v1"
CHUNK_SIZE, CHUNK_OVERLAP = 384, 64

BASE_ENTITIES = {
    "User": "the person speaking in the first person -- I, me, my, myself",
    "Person": "another named individual: a family member, friend, colleague, doctor, teacher",
    "Organization": "a company, employer, school, university, team, club or institution",
    "Location": "a place: a city, country, neighbourhood, venue or building",
    "Activity": "something a person does regularly: a hobby, sport, exercise, practice or routine",
    "Product": "a concrete item or brand someone owns, buys, uses or wants",
    "Event": "a dated happening: a trip, appointment, race, wedding, interview or move",
    "Topic": "a subject of study, skill or interest: a degree, a language, a field of work",
}
BASE_RELATIONS = (
    ("works_for", ("User", "Person"), ("Organization",), "is employed by or works at"),
    ("lives_in", ("User", "Person"), ("Location",), "resides in or has moved to"),
    ("visited", ("User", "Person"), ("Location",), "travelled to or went to"),
    ("practices", ("User", "Person"), ("Activity",), "does this activity, as a hobby or routine"),
    ("owns", ("User", "Person"), ("Product",), "has or possesses this item"),
    ("purchased", ("User", "Person"), ("Product",), "bought or ordered this item"),
    ("prefers", ("User", "Person"), ("Product", "Activity", "Topic"), "likes, favours or chooses"),
    ("studied", ("User", "Person"), ("Topic", "Organization"), "studied, learned or graduated in/from"),
    ("knows", ("User", "Person"), ("Person",), "is related to or acquainted with"),
    ("attended", ("User", "Person"), ("Event",), "took part in or was present at"),
    ("located_in", ("Organization", "Event"), ("Location",), "is situated or takes place in"),
)
VALUE_ENTITIES = {
    "Duration": "a length of time: 25:50, 45 minutes, three weeks, 10-12 hours",
    "Quantity": "a count or amount of things: 3, four, 38 subjects, 5 pairs, 220 pages",
    "Money": "an amount of money: $185, $2,500, $400,000",
    "Date": "a calendar date or day: February 14th, Friday, last Thursday, May 6",
    "TimeWindow": "a span of clock time: 8 am - 4 pm, 2 AM, the day shift",
}
VALUE_RELATIONS = (
    ("personal_best", ("User", "Person"), ("Duration",), "best recorded time or result achieved"),
    ("lasted", ("Event", "Activity"), ("Duration",), "took or went on for this much time"),
    ("spent_time", ("User", "Person"), ("Duration",), "spent this much time on something"),
    ("paid", ("User", "Person"), ("Money",), "paid or spent this amount"),
    ("costs", ("Product", "Event", "Activity"), ("Money",), "has this price"),
    ("owns_count", ("User", "Person"), ("Quantity",), "has or needs this many of something"),
    ("happened_on", ("Event", "Activity"), ("Date",), "took place on this date"),
    ("works_shift", ("User", "Person"), ("TimeWindow",), "is scheduled for this shift"),
)
ARMS = {
    "baseline": (BASE_ENTITIES, BASE_RELATIONS),
    "values": ({**BASE_ENTITIES, **VALUE_ENTITIES}, BASE_RELATIONS + VALUE_RELATIONS),
}


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


def combined_text(turns):
    unique = {}
    for turn in turns:
        text = f"{turn['role']}: {turn['content']}".strip()
        if text:
            unique.setdefault(hashlib.sha256(text.encode()).hexdigest(), text)
    return "\n\n".join(unique.values())


def build_schema(engine, entities, relations):
    schema = engine.create_schema()
    for label, description in entities.items():
        schema = schema.entity(label, description)
    for label, head, tail, description in relations:
        schema = schema.relation(label, head, tail, description)
    return schema


def run(engine, config_cls, text, schema):
    from gliner2.inference.chunking import split_text_into_chunks
    from gliner2.processing.word_splitter import word_splitter_from

    config = config_cls(include_spans=True, include_confidence=True)
    windows = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP, word_splitter=word_splitter_from(engine))
    edges, mentions, infeasible = [], [], 0
    for window in windows:
        joint = engine.extract(window.text, schema, config=config)
        infeasible += 0 if joint.feasible else 1
        by_id = {e.id: e for e in joint.entities}
        for e in joint.entities:
            mentions.append({"type": e.type, "text": e.text, "start": e.start + window.start_char})
        for r in joint.relations:
            head, tail = by_id.get(r.head), by_id.get(r.tail)
            if head is None or tail is None:
                continue
            edges.append(
                {
                    "relation": r.type,
                    "head_type": head.type,
                    "head_text": head.text,
                    "head_start": head.start + window.start_char,
                    "tail_type": tail.type,
                    "tail_text": tail.text,
                    "tail_start": tail.start + window.start_char,
                    "confidence": r.confidence,
                }
            )
    return {"edges": edges, "mentions": mentions, "windows": len(windows), "infeasible": infeasible}


def main():
    sample = json.loads((HERE / "sample_sessions.json").read_text())
    wanted_q = {q["question_id"] for q in sample}
    wanted_s = {s["session_id"] for q in sample for s in q["sessions"]}
    meta = {s["session_id"]: q for q in sample for s in q["sessions"]}
    texts = {}
    for record in stream_records(CORPUS):
        if record["question_id"] not in wanted_q:
            continue
        for sid, turns in zip(record["haystack_session_ids"], record["haystack_sessions"], strict=True):
            if sid in wanted_s:
                texts[sid] = combined_text(turns)
        if len(texts) == len(wanted_s):
            break
    sample_text = {s["session_id"]: s["text"] for q in sample for s in q["sessions"]}
    assert all(texts[s] == sample_text[s] for s in texts), "document drift vs sample"

    from gliner2.joint_ie import JointIE, JointIEConfig

    t0 = time.perf_counter()
    engine = JointIE.from_pretrained(MODEL)
    print(f"loaded {MODEL} in {time.perf_counter() - t0:.1f}s on {engine.device}")
    print(
        f"baseline {len(BASE_ENTITIES)} types/{len(BASE_RELATIONS)} rels; "
        f"values {len(ARMS['values'][0])} types/{len(ARMS['values'][1])} rels\n",
        flush=True,
    )

    # One schema per arm, built once and held for the whole run. gliner2 2.0.0
    # caches compiled schemas under json.dumps(schema, default=repr), and a
    # JointSchema is not JSON-serialisable, so the key is its repr -- a memory
    # address. A schema built per call is freed, the next one can reuse the
    # address, and the engine silently serves the previous compilation.
    schemas = {arm: build_schema(engine, *spec) for arm, spec in ARMS.items()}
    rows = []
    for sid, text in texts.items():
        q = meta[sid]
        print(f"[{q['question_type']}] {sid}")
        row = {
            "session_id": sid,
            "question_type": q["question_type"],
            "question": q["question"],
            "answer": q["answer"],
            "arms": {},
        }
        for arm in ARMS:
            s0 = time.perf_counter()
            row["arms"][arm] = d = run(engine, JointIEConfig, text, schemas[arm])
            print(
                f"  {arm:<9} {d['windows']:>3}w {len(d['edges']):>4} edges {len(d['mentions']):>4} mentions "
                f"infeasible={d['infeasible']} {time.perf_counter() - s0:5.1f}s",
                flush=True,
            )
        rows.append(row)
    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    value_rels = {r[0] for r in VALUE_RELATIONS}
    leaked = sum(e["relation"] in value_rels for r in rows for e in r["arms"]["baseline"]["edges"])
    base_raw = sum(len(r["arms"]["baseline"]["edges"]) for r in rows)
    print(
        f"\nGUARD: baseline raw edges {base_raw} (#350 constrained arm: 1004); value relations leaked into baseline: {leaked}"
    )
    assert leaked == 0 and base_raw == 1004, "arms contaminated"
    report(rows)
    return 0


def report(rows):
    VT = set(VALUE_ENTITIES)
    VR = {r[0] for r in VALUE_RELATIONS}
    print("\n" + "=" * 96 + "\n1. DOES THE ANSWER LAND\n" + "=" * 96)
    seen_q = set()
    for row in rows:
        tokens = {t for t in re.findall(r"[a-z0-9:$,']+", str(row["answer"]).lower()) if len(t) > 1}
        tokens -= {"or", "and", "the", "was", "to", "on", "is", "also"}
        print(f"\n[{row['question_type']}] {row['question'][:80]}  ({row['session_id']})")
        if row["question"] not in seen_q:
            print(f"  answer {str(row['answer'])[:80]!r}")
            seen_q.add(row["question"])
        for arm in ARMS:
            hits = set()
            for e in row["arms"][arm]["edges"]:
                blob = f"{norm(e['head_text'])} {norm(e['tail_text'])}"
                if any(t in blob.split() or (len(t) > 3 and t in blob) for t in tokens):
                    hits.add(
                        f"{e['relation']}({e['head_type']}:{e['head_text']!r} -> {e['tail_type']}:{e['tail_text']!r})"
                    )
            vm = [
                m
                for m in row["arms"][arm]["mentions"]
                if any(t in norm(m["text"]).split() or (len(t) > 3 and t in norm(m["text"])) for t in tokens)
            ]
            print(
                f"    {arm:<9} {len(hits)} answer-bearing edge(s); answer-bearing mentions: "
                f"{sorted({(m['type'], m['text']) for m in vm})[:6]}"
            )
            for h in sorted(hits)[:6]:
                print(f"        {h}")

    print("\n" + "=" * 96 + "\n2. DO THE ORIGINAL 11 RELATIONS DEGRADE\n" + "=" * 96)
    b, v = Counter(), Counter()
    bd, vd = set(), set()
    for row in rows:
        for e in row["arms"]["baseline"]["edges"]:
            b[e["relation"]] += 1
            bd.add((row["session_id"], e["relation"], norm(e["head_text"]), norm(e["tail_text"])))
        for e in row["arms"]["values"]["edges"]:
            if e["relation"] not in VR:
                v[e["relation"]] += 1
                vd.add((row["session_id"], e["relation"], norm(e["head_text"]), norm(e["tail_text"])))
    print(f"  raw: baseline {sum(b.values())} -> values arm {sum(v.values())}")
    print(
        f"  distinct claims: baseline {len(bd)} -> values arm {len(vd)}; shared {len(bd & vd)}, "
        f"lost {len(bd - vd)}, gained {len(vd - bd)}"
    )
    for rel in sorted(set(b) | set(v)):
        print(f"    {rel:<14}{b[rel]:>6}{v[rel]:>6}{v[rel] - b[rel]:>+6}")
    print("  sample of lost claims:")
    for c in sorted(bd - vd)[:12]:
        print(f"      {c[1]}({c[2]!r} -> {c[3]!r})")

    print("\n" + "=" * 96 + "\n3. WHAT THE VALUE TYPES EXTRACTED\n" + "=" * 96)
    vm, ve = Counter(), Counter()
    ex = defaultdict(Counter)
    vedges = []
    for row in rows:
        for m in row["arms"]["values"]["mentions"]:
            if m["type"] in VT:
                vm[m["type"]] += 1
                ex[m["type"]][norm(m["text"])] += 1
        for e in row["arms"]["values"]["edges"]:
            if e["relation"] in VR:
                ve[e["relation"]] += 1
                vedges.append(e)
    print(f"  value mentions {sum(vm.values())}, value edges {sum(ve.values())}")
    for t, n in vm.most_common():
        print(f"    {t:<11}{n:>5}  {ex[t].most_common(8)}")
    print(f"  value relations fired: {dict(ve.most_common())}")
    print("  distinct value edges:")
    for s in sorted(
        {f"{e['relation']}({e['head_type']}:{e['head_text']!r} -> {e['tail_type']}:{e['tail_text']!r})" for e in vedges}
    )[:60]:
        print(f"      {s}")

    print("\n" + "=" * 96 + "\n4. VALUE-NODE COLLISIONS WITHIN A SESSION\n" + "=" * 96)
    total = coll = 0
    worst = []
    for row in rows:
        keys = defaultdict(set)
        for m in row["arms"]["values"]["mentions"]:
            if m["type"] in VT:
                keys[(m["type"], norm(m["text"]))].add(m["start"])
        for (t, x), spans in keys.items():
            total += 1
            if len(spans) > 1:
                coll += 1
                worst.append((len(spans), t, x))
    print(f"  distinct (type,text) keys {total}; covering >1 span {coll} ({100 * coll / max(total, 1):.0f}%)")
    for n, t, x in sorted(worst, reverse=True)[:12]:
        print(f"      {t}:{x!r} x{n}")


if __name__ == "__main__":
    sys.exit(main())
