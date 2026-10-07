"""GLiNER2Observer: the tables derivation prunes from, with the engine stood in for."""

from typing import Any, cast

from hygm import HygmModel, NodeType, RelationType
from unstructured2graph import Document, Segment
from unstructured2graph.gliner2_observer import GLiNER2Observer

from .gliner2_fakes import FakeEngine

MODEL = HygmModel(
    node_types=(
        NodeType("User", "the user", "global"),
        NodeType("Person", "someone else", "global"),
        NodeType("Location", "a place", "global"),
        NodeType("Quantity", "a count", "span"),
    ),
    relation_types=(RelationType("visited"), RelationType("owns_count", end_labels=("Quantity",))),
)


def _session(*turns, user_id="u1"):
    text, segments = "", []
    for role, turn in turns:
        segments.append(Segment(len(text), len(text) + len(turn), role))
        text += turn
    return Document(text=text, segments=tuple(segments), user_id=user_id)


def test_tables_count_types_as_resolved_and_pairs_with_examples():
    engine = FakeEngine(
        surfaces={"I": "User", "Paris": "Location", "Kahlo": "User"},
        relations=[("visited", "I", "Paris", 0.9), ("visited", "Kahlo", "Paris", 0.8)],
    )
    sample = [
        _session(("user", "I visited Paris"), ("assistant", "Kahlo visited Paris. I agree.")),
        _session(("user", "Paris again")),
    ]

    tables = GLiNER2Observer(model=engine).observe(MODEL, sample)

    visited = tables["relations"]["visited"]
    assert visited["edges"] == 2
    assert {(p["head"], p["tail"], p["count"]) for p in visited["pairs"]} == {
        ("User", "Location", 1),
        ("Person", "Location", 1),
    }
    assert {e for p in visited["pairs"] for e in p["examples"]} == {"I -> Paris", "Kahlo -> Paris"}
    assert tables["types"]["User"]["mentions"] == 1  # the assistant's "I" is dropped, Kahlo is a Person
    assert tables["types"]["Person"]["top_texts"] == ["kahlo"]
    paris = tables["types"]["Location"]
    assert (paris["mentions"], paris["sessions"], paris["texts_recurring_across_sessions"]) == (3, 2, 1.0)
    assert tables["relations"]["owns_count"] == {"edges": 0, "pairs": []}
    assert tables["types"]["Quantity"]["mentions"] == 0


def test_the_value_pass_shapes_what_is_observed():
    engine = FakeEngine(
        surfaces={"I": "User", "3 cats": "Location"},
        value_surfaces={"I": "User", "3": "Quantity"},
        value_relations=[("owns_count", "I", "3", 0.9)],
    )

    tables = GLiNER2Observer(model=engine).observe(MODEL, [_session(("user", "I have 3 cats"))])

    assert tables["types"]["Quantity"]["mentions"] == 1
    assert tables["types"]["Location"]["mentions"] == 0
    assert tables["relations"]["owns_count"]["pairs"][0]["head"] == "User"


def test_the_engine_is_reused_across_candidates():
    engine = FakeEngine()
    observer = GLiNER2Observer(model=engine)

    observer.observe(MODEL, [])
    observer.observe(MODEL, [])

    assert observer._model is engine
    assert len(engine.compiled) == 4  # each candidate's schema and value schema, one engine


def test_measure_scores_catch_all_share_and_user_turn_coverage():
    model = HygmModel(
        node_types=(*MODEL.node_types, NodeType("Topic", "a subject", "chunk")),
        relation_types=MODEL.relation_types,
    )
    engine = FakeEngine(
        surfaces={"I": "User", "Paris": "Location", "jazz": "Topic"},
        relations=[("visited", "I", "Paris", 0.9)],
    )
    sample = [_session(("user", "I visited Paris"), ("assistant", "jazz there?"), ("user", "I like jazz"))]

    measured = GLiNER2Observer(model=engine).measure(model, sample, catch_alls=("Topic",))

    assert (measured.mentions, measured.user_turns) == (5, 2)
    assert measured.catch_all_share == 2 / 5
    assert measured.coverage == 1 / 2  # only the first user turn yields a typed relation
    assert measured.spans[(0, 10, 15)] == "Location"


def test_a_window_budget_reads_an_even_spread_and_measures_only_what_it_read():
    from unstructured2graph.gliner2_observer import _spread

    assert _spread(list(range(10)), 4) == [0, 3, 6, 9]
    assert _spread(list(range(3)), 4) == [0, 1, 2]
    assert _spread(list(range(10)), None) == list(range(10))

    engine = FakeEngine(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.9)])
    sample = [_session(("user", f"I visited Paris {i}"), ("assistant", "nice")) for i in range(5)]

    measured = GLiNER2Observer(model=engine, window_budget=3).measure(MODEL, sample, catch_alls=())

    main_schema = engine.compiled[0]  # the main pass's; the value-only pass compiles second
    assert sum(schema[1] is main_schema for _, schema, _ in cast("list[Any]", engine.calls)) == 3  # 3 of 10 windows
    assert measured.user_turns == 2  # windows 0, 4 (user turns) and 9 (an assistant turn)
    assert measured.coverage == 1.0
