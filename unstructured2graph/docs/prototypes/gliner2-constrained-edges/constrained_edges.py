"""#350 prototype: wire a constrained JointSchema ontology and read the edges.

Throwaway. Not a change to GLiNER2Backend, and nothing here is meant to ship --
it exists so the typed relation model can be judged against real session text
before the plan is declared ready.

What it does, per #350 and its briefs:

1. Declares a hand-written stand-in vocabulary with domain/range bindings
   (#347 decided real vocabularies are derived per corpus; a stand-in is a fine
   proxy for judging *extraction*, and waiting on #353 would serialise two
   independent questions).
2. Wires it through `gliner2.joint_ie.JointSchema` via `JointIE` -- the real
   entry point #345 established. `AutoExtractor.extract()` and
   `compile_schema().build()` silently drop all typing.
3. Drives the windows itself instead of calling `extract_long_text`, because
   that helper omits `feasible` when it builds its merged result
   (`long_text.py:77`), swallowing per-window infeasibility -- which #348 needs
   in order to retry an infeasible window permissively.
4. Runs every session twice: once with the vocabulary's real
   `start_labels`/`end_labels`, once with permissive endpoints (all declared
   entity types -- the only way #345 found to express "unconstrained", and
   exactly the translation the plan's `start_labels=()` compat rule needs).
   Diffing the two arms is the honest read on what constrained decoding buys.
5. Prints the edges as readable triples for human judgment -- the map's stated
   decision criterion -- and reports identity/fragmentation shape under #346's
   exact-normalized-text-within-type rule.

Usage:
    .venv/bin/python constrained_edges.py [--windows-per-session N] [--sessions N]
"""

import argparse
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).parent
SAMPLE = HERE / "sample_sessions.json"
MODEL = "fastino/gliner2.5-base-v1"

# GLiNER2Backend's own defaults, so windows match what production would see.
CHUNK_SIZE = 384
CHUNK_OVERLAP = 64


# --------------------------------------------------------------------------
# The stand-in vocabulary
#
# Shaped by what #347 measured about this corpus: 98 of 100 questions are
# first-person, so `User` is an entity type and most relations are user-headed;
# 42% turn on time or change, so every edge is stamped with the session's date
# (`valid_at`) by the writer rather than extracted.
#
# Deliberately small. Per #345 an endpoint pair is expanded into candidates as
# head_types x tail_types, so the permissive arm costs |types|^2 per relation
# type -- 8 entity types is already 64 pairs per relation against 1-3 typed.
# --------------------------------------------------------------------------

ENTITY_TYPES: dict[str, str] = {
    "User": "the person speaking in the first person -- I, me, my, myself",
    "Person": "another named individual: a family member, friend, colleague, doctor, teacher",
    "Organization": "a company, employer, school, university, team, club or institution",
    "Location": "a place: a city, country, neighbourhood, venue or building",
    "Activity": "something a person does regularly: a hobby, sport, exercise, practice or routine",
    "Product": "a concrete item or brand someone owns, buys, uses or wants",
    "Event": "a dated happening: a trip, appointment, race, wedding, interview or move",
    "Topic": "a subject of study, skill or interest: a degree, a language, a field of work",
}


@dataclass(frozen=True)
class Relation:
    """One relation type with the plan's `start_labels`/`end_labels` binding."""

    label: str
    start_labels: tuple[str, ...]
    end_labels: tuple[str, ...]
    description: str


RELATION_TYPES: tuple[Relation, ...] = (
    Relation("works_for", ("User", "Person"), ("Organization",), "is employed by or works at"),
    Relation("lives_in", ("User", "Person"), ("Location",), "resides in or has moved to"),
    Relation("visited", ("User", "Person"), ("Location",), "travelled to or went to"),
    Relation("practices", ("User", "Person"), ("Activity",), "does this activity, as a hobby or routine"),
    Relation("owns", ("User", "Person"), ("Product",), "has or possesses this item"),
    Relation("purchased", ("User", "Person"), ("Product",), "bought or ordered this item"),
    Relation("prefers", ("User", "Person"), ("Product", "Activity", "Topic"), "likes, favours or chooses"),
    Relation("studied", ("User", "Person"), ("Topic", "Organization"), "studied, learned or graduated in/from"),
    Relation("knows", ("User", "Person"), ("Person",), "is related to or acquainted with"),
    Relation("attended", ("User", "Person"), ("Event",), "took part in or was present at"),
    Relation("located_in", ("Organization", "Event"), ("Location",), "is situated or takes place in"),
)

ALL_TYPES: tuple[str, ...] = tuple(ENTITY_TYPES)

#: #346's rule is per-type opt-in: identical surface text means the same entity
#: only for types where that actually holds. Name-like types opt in; `Topic`
#: and `Activity` are the ones most likely to build hubs, and are the closest
#: thing in this stand-in to the `Concept` type #346 measured (12,583 nodes).
MERGEABLE_TYPES: frozenset[str] = frozenset({"User", "Person", "Organization", "Location", "Product", "Event"})


def normalize(text: str) -> str:
    """GLiNER2Backend._normalize_text, verbatim."""
    return " ".join(text.strip().lower().split())


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


@dataclass
class WindowRun:
    index: int
    start_char: int
    chars: int
    feasible: bool
    entities: int
    relations: int
    seconds: float


@dataclass
class Edge:
    """One extracted relation, endpoints resolved to document coordinates."""

    relation: str
    head_type: str
    head_text: str
    head_span: tuple[int, int]
    tail_type: str
    tail_text: str
    tail_span: tuple[int, int]
    confidence: float | None
    window: int
    session_id: str
    valid_at: str  # stamped by the writer from the session date, never extracted

    def key(self) -> tuple[str, str, str]:
        """Identity for diffing arms: the claim, ignoring where it was found."""
        return (self.relation, normalize(self.head_text), normalize(self.tail_text))

    def typed_key(self) -> tuple[str, str, str, str, str]:
        return (self.relation, self.head_type, normalize(self.head_text), self.tail_type, normalize(self.tail_text))

    def triple(self) -> str:
        return f"{self.relation}({self.head_type}:{self.head_text!r} -> {self.tail_type}:{self.tail_text!r})"


@dataclass
class ArmResult:
    arm: str
    windows: list[WindowRun] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    mentions: list[tuple[str, str, int, int]] = field(default_factory=list)  # (type, text, start, end)
    seconds: float = 0.0

    @property
    def infeasible(self) -> list[WindowRun]:
        return [w for w in self.windows if not w.feasible]


def build_schema(engine, permissive: bool):
    """The same specification compiled two ways -- #348's two compilations, in
    one arm each. Permissive is "all declared entity types", because #345
    verified empty/None/"" endpoints all raise; that is also the translation the
    plan's `start_labels=()` rule needs on our side.
    """
    schema = engine.create_schema()
    for label, description in ENTITY_TYPES.items():
        schema = schema.entity(label, description)
    for relation in RELATION_TYPES:
        head = ALL_TYPES if permissive else relation.start_labels
        tail = ALL_TYPES if permissive else relation.end_labels
        schema = schema.relation(relation.label, head, tail, relation.description)
    return schema


def run_arm(engine, config_cls, session: dict, *, permissive: bool, max_windows: int | None) -> ArmResult:
    """Extract one session window by window, keeping per-window `feasible`."""
    from gliner2.inference.chunking import split_text_into_chunks
    from gliner2.processing.word_splitter import word_splitter_from

    text = session["text"]
    schema = build_schema(engine, permissive)
    config = config_cls(include_spans=True, include_confidence=True)

    windows = split_text_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP, word_splitter=word_splitter_from(engine))
    if max_windows is not None:
        windows = windows[:max_windows]

    result = ArmResult(arm="permissive" if permissive else "constrained")
    started = time.perf_counter()
    for index, window in enumerate(windows):
        window_started = time.perf_counter()
        joint = engine.extract(window.text, schema, config=config)
        elapsed = time.perf_counter() - window_started

        by_id = {e.id: e for e in joint.entities}
        for entity in joint.entities:
            result.mentions.append(
                (entity.type, entity.text, entity.start + window.start_char, entity.end + window.start_char)
            )
        for relation in joint.relations:
            head, tail = by_id.get(relation.head), by_id.get(relation.tail)
            if head is None or tail is None:  # cannot happen per #345, asserted by omission
                continue
            result.edges.append(
                Edge(
                    relation=relation.type,
                    head_type=head.type,
                    head_text=head.text,
                    head_span=(head.start + window.start_char, head.end + window.start_char),
                    tail_type=tail.type,
                    tail_text=tail.text,
                    tail_span=(tail.start + window.start_char, tail.end + window.start_char),
                    confidence=relation.confidence,
                    window=index,
                    session_id=session["session_id"],
                    valid_at=session["date"],
                )
            )
        result.windows.append(
            WindowRun(
                index=index,
                start_char=window.start_char,
                chars=len(window.text),
                feasible=bool(joint.feasible),
                entities=len(joint.entities),
                relations=len(joint.relations),
                seconds=elapsed,
            )
        )
        print(
            f"    window {index:>3} {len(window.text):>5}ch  "
            f"{'FEASIBLE' if joint.feasible else 'INFEASIBLE':<10} "
            f"{len(joint.entities):>3}e {len(joint.relations):>3}r  {elapsed:5.1f}s",
            flush=True,
        )
    result.seconds = time.perf_counter() - started
    return result


# --------------------------------------------------------------------------
# Reads
# --------------------------------------------------------------------------


def dedupe_edges(edges: list[Edge]) -> dict[tuple[str, str, str, str, str], list[Edge]]:
    """Collapse mention-level duplicates the way #346's identity rule would.

    Duplication is expected and was verified in #345: two mentions of one
    organisation become two entities and duplicate every edge. Grouping by
    typed_key() is what the merge at write time does for opted-in types.
    """
    grouped: dict[tuple[str, str, str, str, str], list[Edge]] = defaultdict(list)
    for edge in edges:
        grouped[edge.typed_key()].append(edge)
    return grouped


def identity_shape(result: ArmResult) -> dict:
    """#346's read: fragmentation under chunk-scoped vs merged identity."""
    per_type_mentions: Counter[str] = Counter()
    chunk_scoped: set[tuple[str, str, int]] = set()  # (type, normalized text, window-ish: span start)
    merged: set[tuple[str, str]] = set()
    mention_counts: Counter[tuple[str, str]] = Counter()
    for entity_type, text, start, _end in result.mentions:
        normalized = normalize(text)
        if not normalized:
            continue
        per_type_mentions[entity_type] += 1
        chunk_scoped.add((entity_type, normalized, start))
        merged.add((entity_type, normalized))
        mention_counts[(entity_type, normalized)] += 1

    degree: Counter[tuple[str, str]] = Counter()
    for edge in result.edges:
        degree[(edge.head_type, normalize(edge.head_text))] += 1
        degree[(edge.tail_type, normalize(edge.tail_text))] += 1

    return {
        "mentions_total": sum(per_type_mentions.values()),
        "mentions_per_type": dict(per_type_mentions.most_common()),
        "nodes_span_scoped": len(chunk_scoped),
        "nodes_merged": len(merged),
        "nodes_merged_mergeable_types_only": len({k for k in merged if k[0] in MERGEABLE_TYPES}),
        "top_mention_counts": [{"type": t, "text": x, "mentions": n} for (t, x), n in mention_counts.most_common(12)],
        "top_degree_merged": [{"type": t, "text": x, "degree": n} for (t, x), n in degree.most_common(12)],
    }


def report_session(question: dict, session: dict, arms: dict[str, ArmResult]) -> dict:
    """Print the triples for human judgment and return the machine-readable row."""
    constrained, permissive = arms["constrained"], arms["permissive"]

    print()
    print(f"  --- {session['session_id']} ({session['date']}, {len(session['text'])} chars) ---")
    for arm_name, arm in (("constrained", constrained), ("permissive", permissive)):
        grouped = dedupe_edges(arm.edges)
        print(
            f"  {arm_name}: {len(arm.edges)} raw edges, {len(grouped)} distinct after merge, "
            f"{len(arm.infeasible)}/{len(arm.windows)} windows infeasible, {arm.seconds:.0f}s"
        )
        for key, members in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][2])):
            relation, head_type, head_text, tail_type, tail_text = key
            confidences = [e.confidence for e in members if e.confidence is not None]
            confidence = f"{max(confidences):.2f}" if confidences else "n/a"
            print(
                f"      {relation:<12} ({head_type}) {head_text!r} -> ({tail_type}) {tail_text!r}"
                f"   x{len(members)} conf<={confidence} valid_at={members[0].valid_at}"
            )

    constrained_keys = {e.key() for e in constrained.edges}
    permissive_keys = {e.key() for e in permissive.edges}
    only_constrained = sorted(constrained_keys - permissive_keys)
    only_permissive = sorted(permissive_keys - constrained_keys)

    # A claim can survive into both arms while its *endpoint typing* changes --
    # the constrained arm cannot type an endpoint outside the declared binding,
    # so the same triple comes back with a different head/tail type. That is a
    # third outcome besides "kept" and "suppressed", and it is invisible to a
    # diff keyed on text alone.
    constrained_typing = {e.key(): (e.head_type, e.tail_type) for e in constrained.edges}
    permissive_typing = {e.key(): (e.head_type, e.tail_type) for e in permissive.edges}
    retyped = [
        {
            "claim": list(key),
            "permissive_types": list(permissive_typing[key]),
            "constrained_types": list(constrained_typing[key]),
        }
        for key in sorted(constrained_keys & permissive_keys)
        if constrained_typing[key] != permissive_typing[key]
    ]

    print(
        f"  diff: {len(constrained_keys & permissive_keys)} shared claims, "
        f"{len(only_constrained)} constrained-only, {len(only_permissive)} permissive-only, "
        f"{len(retyped)} shared-but-retyped"
    )
    for item in retyped[:10]:
        relation, head, tail = item["claim"]
        p_head, p_tail = item["permissive_types"]
        c_head, c_tail = item["constrained_types"]
        print(
            f"      ~ {relation}({head!r} -> {tail!r}): ({p_head},{p_tail}) permissive -> ({c_head},{c_tail}) constrained"
        )
    if only_constrained:
        print("    constrained-only (beam reallocation, if #348's claim holds):")
        for relation, head, tail in only_constrained[:15]:
            print(f"      + {relation}({head!r} -> {tail!r})")
    if only_permissive:
        print("    permissive-only (suppressed by the domain/range constraint):")
        for relation, head, tail in only_permissive[:15]:
            print(f"      - {relation}({head!r} -> {tail!r})")

    # A permissive edge whose endpoint types violate the declared binding is
    # exactly what the post-hoc Cypher check in #348 would flag.
    binding = {r.label: (set(r.start_labels), set(r.end_labels)) for r in RELATION_TYPES}
    violations = [
        e
        for e in permissive.edges
        if e.head_type not in binding[e.relation][0] or e.tail_type not in binding[e.relation][1]
    ]

    return {
        "question_id": question["question_id"],
        "question_type": question["question_type"],
        "question": question["question"],
        "answer": question["answer"],
        "session_id": session["session_id"],
        "session_date": session["date"],
        "session_chars": len(session["text"]),
        "arms": {
            name: {
                "windows": [vars(w) for w in arm.windows],
                "infeasible_windows": len(arm.infeasible),
                "raw_edges": len(arm.edges),
                "distinct_edges": len(dedupe_edges(arm.edges)),
                "seconds": arm.seconds,
                "identity": identity_shape(arm),
                "edges": [vars(e) for e in arm.edges],
            }
            for name, arm in arms.items()
        },
        "diff": {
            "shared": sorted(constrained_keys & permissive_keys),
            "only_constrained": only_constrained,
            "only_permissive": only_permissive,
            "shared_but_retyped": retyped,
        },
        "permissive_domain_range_violations": [
            {"triple": e.triple(), "relation": e.relation, "head_type": e.head_type, "tail_type": e.tail_type}
            for e in violations
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=None, help="cap total sessions processed")
    parser.add_argument("--windows-per-session", type=int, default=None, help="cap windows per session (smoke test)")
    parser.add_argument("--out", type=Path, default=HERE / "constrained_edges_output.json")
    args = parser.parse_args()

    from gliner2.joint_ie import JointIE, JointIEConfig

    questions = json.loads(SAMPLE.read_text(encoding="utf-8"))
    pairs = [(q, s) for q in questions for s in q["sessions"]]
    if args.sessions is not None:
        pairs = pairs[: args.sessions]

    print(f"loading {MODEL} ...", flush=True)
    started = time.perf_counter()
    engine = JointIE.from_pretrained(MODEL)
    print(f"loaded in {time.perf_counter() - started:.1f}s on {engine.device}", flush=True)
    print(
        f"vocabulary: {len(ENTITY_TYPES)} entity types, {len(RELATION_TYPES)} relation types; "
        f"permissive arm expands every endpoint to all {len(ALL_TYPES)}",
        flush=True,
    )

    rows = []
    for question, session in pairs:
        print()
        print("=" * 100)
        print(f"[{question['question_type']}] {question['question']}")
        print(f"  expected answer: {question['answer']!r}")
        arms = {}
        for permissive in (False, True):
            name = "permissive" if permissive else "constrained"
            print(f"  {name}:")
            arms[name] = run_arm(
                engine, JointIEConfig, session, permissive=permissive, max_windows=args.windows_per_session
            )
        rows.append(report_session(question, session, arms))
        args.out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    print()
    print("=" * 100)
    print("TOTALS")
    for name in ("constrained", "permissive"):
        raw = sum(r["arms"][name]["raw_edges"] for r in rows)
        distinct = sum(r["arms"][name]["distinct_edges"] for r in rows)
        infeasible = sum(r["arms"][name]["infeasible_windows"] for r in rows)
        windows = sum(len(r["arms"][name]["windows"]) for r in rows)
        seconds = sum(r["arms"][name]["seconds"] for r in rows)
        print(
            f"  {name:<12} {raw:>5} raw edges  {distinct:>5} distinct  "
            f"{infeasible:>3}/{windows} windows infeasible  {seconds:>6.0f}s"
        )
    shared = sum(len(r["diff"]["shared"]) for r in rows)
    only_c = sum(len(r["diff"]["only_constrained"]) for r in rows)
    only_p = sum(len(r["diff"]["only_permissive"]) for r in rows)
    violations = sum(len(r["permissive_domain_range_violations"]) for r in rows)
    retyped = sum(len(r["diff"]["shared_but_retyped"]) for r in rows)
    print(f"  diff: {shared} shared ({retyped} of them retyped), {only_c} constrained-only, {only_p} permissive-only")
    print(f"  permissive edges violating the declared domain/range: {violations}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
