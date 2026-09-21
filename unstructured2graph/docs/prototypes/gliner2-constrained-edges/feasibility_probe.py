"""Is `feasible=False` reachable with a plausibly-tight ontology? (#350's brief)

#348 chose to retry an infeasible window with permissive endpoints and write the
result flagged. That path only exists if a window can actually come back
infeasible. The full run saw 0/109 in both arms, so this probes progressively
tighter schemas over the same windows.

`JointResult.feasible` is False only when *every* candidate assignment fails
`validate_solution` (optimizers/beam.py:96-109). Typing and cardinality are
prohibitive -- the decoder satisfies them by dropping edges, and the empty
solution always validates -- so the hypothesis under test is that only
completeness-style constraints (symmetric/inverse companions, which are injected
*after* screening) can make a window infeasible.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
MODEL = "fastino/gliner2.5-base-v1"
CHUNK_SIZE, CHUNK_OVERLAP = 384, 64
WINDOWS = 6

ENTITIES = {
    "User": "the person speaking in the first person -- I, me, my, myself",
    "Person": "another named individual: a family member, friend, colleague, doctor, teacher",
    "Organization": "a company, employer, school, university, team, club or institution",
    "Location": "a place: a city, country, neighbourhood, venue or building",
    "Product": "a concrete item or brand someone owns, buys, uses or wants",
}


def schema_typed(engine):
    """A: domain/range only -- the constrained arm of the main run."""
    s = engine.create_schema()
    for label, description in ENTITIES.items():
        s = s.entity(label, description)
    s = s.relation("works_for", ("User", "Person"), ("Organization",), "is employed by")
    s = s.relation("lives_in", ("User", "Person"), ("Location",), "resides in")
    s = s.relation("owns", ("User", "Person"), ("Product",), "possesses")
    s = s.relation("knows", ("User", "Person"), ("Person",), "is acquainted with")
    return s


def schema_cardinality(engine):
    """B: + hard cardinality and no self-loops -- prohibitive constraints only."""
    s = engine.create_schema()
    for label, description in ENTITIES.items():
        s = s.entity(label, description)
    s = s.relation("works_for", ("User", "Person"), ("Organization",), "is employed by", max_per_head=1, max_per_tail=1)
    s = s.relation("lives_in", ("User", "Person"), ("Location",), "resides in", max_per_head=1, max_per_tail=1)
    s = s.relation("owns", ("User", "Person"), ("Product",), "possesses", max_per_head=1, max_per_tail=1)
    s = s.relation("knows", ("User", "Person"), ("Person",), "is acquainted with", max_per_head=1, max_per_tail=1)
    return s.no_self_loops().at_most(per_head=1)


def schema_completeness(engine):
    """C: + symmetric/inverse companions against a cardinality cap.

    `knows` symmetric means every accepted edge injects its reverse *after*
    screening, so the companion can push a head past max_per_head=1 -- a
    violation the incremental checks never saw.
    """
    s = engine.create_schema()
    for label, description in ENTITIES.items():
        s = s.entity(label, description)
    # Symmetric endpoints must be type-compatible -- the library rejects
    # ("User","Person") -> ("Person",) outright, which is itself a constraint on
    # the plan's start_labels/end_labels model: a symmetric relation type forces
    # its domain and range to be the same set.
    s = s.relation(
        "knows",
        ("User", "Person"),
        ("User", "Person"),
        "is acquainted with",
        symmetric=True,
        max_per_head=1,
        max_per_tail=1,
    )
    s = s.relation(
        "works_for", ("User", "Person"), ("Organization",), "is employed by", inverse="employs", max_per_head=1
    )
    s = s.relation("employs", ("Organization",), ("User", "Person"), "employs", inverse="works_for", max_per_head=1)
    return s.no_self_loops()


def main() -> int:
    from gliner2.inference.chunking import split_text_into_chunks
    from gliner2.joint_ie import JointIE, JointIEConfig
    from gliner2.processing.word_splitter import word_splitter_from

    questions = json.loads((HERE / "sample_sessions.json").read_text())
    sessions = [s for q in questions for s in q["sessions"]][:2]

    engine = JointIE.from_pretrained(MODEL)
    config = JointIEConfig(include_spans=True, include_confidence=True)

    for name, builder in (
        ("A typed-only", schema_typed),
        ("B + cardinality (prohibitive)", schema_cardinality),
        ("C + symmetric/inverse (completeness)", schema_completeness),
    ):
        try:
            schema = builder(engine)
        except Exception as exc:  # a schema the library refuses to build is itself a finding
            print(f"{name:<40} schema rejected: {type(exc).__name__}: {exc}")
            continue
        infeasible = total = edges = 0
        for session in sessions:
            windows = split_text_into_chunks(
                session["text"], CHUNK_SIZE, CHUNK_OVERLAP, word_splitter=word_splitter_from(engine)
            )[:WINDOWS]
            for window in windows:
                result = engine.extract(window.text, schema, config=config)
                total += 1
                edges += len(result.relations)
                if not result.feasible:
                    infeasible += 1
        print(f"{name:<40} {infeasible}/{total} windows infeasible, {edges} relations kept")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
