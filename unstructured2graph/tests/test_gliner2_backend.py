"""GLiNER2Backend against a fake joint engine (tests/gliner2_fakes.py).

Pure logic -- schema, windows, the mention resolver, identity -- is tested
directly. What the backend writes is tested against a real Memgraph (the
`memgraph` fixture skips without one), per the repo's prefer-real-Memgraph
testing policy.
"""

import pytest

from unstructured2graph import Chunk, Document, EntityType, Ontology, RelationType, Segment, from_documents
from unstructured2graph.gliner2_backend import (
    BIND_USER,
    DEFAULT_CANDIDATE_CAP,
    DROP,
    KEEP,
    GLiNER2Backend,
    Mention,
    Resolution,
    _entity_id,
    _normalize_text,
    _source_text,
    _word_windows,
    resolve_user_mentions,
)

from .gliner2_fakes import FakeEngine

ONTOLOGY = Ontology(
    entity_types=(
        EntityType("User", "the person speaking in the first person", "global"),
        EntityType("Person", "another named individual", "global"),
        EntityType("Location", "a place", "global"),
        EntityType("Product", "an item", "chunk"),
        EntityType("Quantity", "a count", "span"),
    ),
    relation_types=(
        RelationType("visited", "travelled to", ("User", "Person"), ("Location",)),
        RelationType("owns_count", "has this many", ("User", "Person"), ("Quantity",)),
        RelationType("mentions"),
    ),
)


def _backend(**engine_kwargs):
    return GLiNER2Backend(ontology=ONTOLOGY, model=FakeEngine(**engine_kwargs))


def _session(*turns, user_id="u1"):
    """A Document shaped like sessions-graph builds one: turns joined by a blank line.

    Each turn is (role, body, when) or (role, body, when, source_id)."""
    text, segments, cursor = "", [], 0
    for role, body, when, *source in turns:
        turn = f"{role}: {body}"
        if text:
            text += "\n\n"
            cursor += 2
        segments.append(Segment(cursor, cursor + len(turn), role, when, *source))
        text += turn
        cursor += len(turn)
    return Document(text=text, segments=tuple(segments), user_id=user_id)


# --- schema and configuration -------------------------------------------------


def test_schema_is_typed_from_the_ontology_and_compiled_once():
    backend = _backend()
    engine = backend.engine
    assert len(engine.compiled) == 2  # the schema, and the value-only one
    schema = engine.compiled[0]
    assert [name for name, _ in schema.entities] == ["User", "Person", "Location", "Product", "Quantity"]
    assert schema.relations[0] == ("visited", ("User", "Person"), ("Location",))
    everything = ("User", "Person", "Location", "Product", "Quantity")
    assert schema.relations[2] == ("mentions", everything, everything)  # JointSchema rejects an empty endpoint


def test_the_value_only_schema_holds_the_value_relations_their_heads_and_tails():
    values = _backend().engine.compiled[1]
    assert [name for name, _ in values.entities] == ["User", "Person", "Quantity"]
    assert values.relations == [("owns_count", ("User", "Person"), ("Quantity",))]


def test_no_value_relation_means_no_value_pass():
    ontology = Ontology(entity_types=ONTOLOGY.entity_types, relation_types=ONTOLOGY.relation_types[:1])
    backend = GLiNER2Backend(ontology=ontology, model=FakeEngine())
    assert len(backend.engine.compiled) == 1


def test_candidate_caps_default_high_enough_that_user_edges_survive():
    config = _backend()._config
    assert config.relation_pair_cap == config.max_edges_per_type == DEFAULT_CANDIDATE_CAP == 4096
    assert config.optimizer == "greedy"
    assert GLiNER2Backend(ontology=ONTOLOGY, model=FakeEngine(), candidate_cap=10)._config.relation_pair_cap == 10


def test_invalid_workspace_raises():
    with pytest.raises(ValueError, match="workspace"):
        GLiNER2Backend(ontology=ONTOLOGY, model=FakeEngine(), workspace="bad-name")


@pytest.mark.asyncio
async def test_every_window_extracts_against_the_one_held_compiled_schema():
    backend = _backend()
    document = _session(("user", "hello", None), ("assistant", "hi", None))
    chunk = Chunk(document.text, "h", document.segments, "u1")
    backend._extract_sync(chunk)
    backend._extract_sync(chunk)
    schemas = {id(schema) for _, schema, _ in backend.engine.calls}
    assert len(schemas) == 2
    assert len(backend.engine.compiled) == 2


# --- windows ---------------------------------------------------------------------


def test_one_window_per_segment():
    backend = _backend()
    document = _session(("user", "I went to Paris", None), ("assistant", "Nice", None))
    windows = backend._windows(Chunk(document.text, "h", document.segments))
    assert [document.text[w.start : w.end] for w in windows] == ["user: I went to Paris", "assistant: Nice"]
    assert [w.segment.role for w in windows] == ["user", "assistant"]


def test_a_segment_longer_than_chunk_size_is_split_with_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    ranges = _word_windows(text, 0, len(text), size=4, overlap=1)
    assert [text[a:b] for a, b in ranges] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert _word_windows(text, 0, len(text), size=10, overlap=1) == [(0, len(text))]


def test_windows_of_a_later_segment_stay_inside_it():
    """Offsets are the chunk's, not the segment's: a later turn's windows must not drift into the next one."""
    first, second, third = "user: hi", "assistant: " + " ".join(f"w{i}" for i in range(10)), "user: bye"
    text = f"{first}\n\n{second}\n\n{third}"
    start = len(first) + 2
    ranges = _word_windows(text, start, start + len(second), size=4, overlap=1)
    assert [text[a:b] for a, b in ranges] == ["assistant: w0 w1", "w1 w2 w3 w4", "w4 w5 w6 w7", "w7 w8 w9"]
    assert all(start <= a < b <= start + len(second) for a, b in ranges)


def test_punctuation_counts_as_a_word_as_gliner2_counts_it():
    assert _word_windows("a, b. c!", 0, 8, size=3, overlap=0) == [(0, 4), (4, 8)]  # "a , b" then ". c !"


def test_without_segments_the_whole_text_is_word_windowed():
    backend = GLiNER2Backend(ontology=ONTOLOGY, model=FakeEngine(), chunk_size=3, chunk_overlap=0)
    windows = backend._windows(Chunk("a b c d e", "h"))
    assert [(w.start, w.end, w.segment) for w in windows] == [(0, 5, None), (6, 9, None)]


# --- the mention resolver -------------------------------------------------------


@pytest.mark.parametrize(
    ("entity_type", "surface", "role", "expected"),
    [
        ("User", "I", "user", BIND_USER),
        ("User", "user", "user", BIND_USER),  # the "user: " role prefix
        ("User", "my", "user", BIND_USER),
        ("User", "you", "assistant", BIND_USER),
        ("User", "I", "assistant", DROP),  # first person in the assistant's mouth
        ("User", "assistant", "assistant", DROP),
        ("User", "you", "user", DROP),  # the user addressing the assistant
        ("User", "I", None, DROP),  # a tool result is nobody's utterance
        ("User", "Kahlo", "user", Resolution("keep", "Person")),  # a third party
        ("User", "Kahlo", "assistant", Resolution("keep", "Person")),
        ("Location", "I", "user", KEEP),  # only User and Person mentions are resolved
        ("Person", "I", "user", BIND_USER),  # the model types pronouns Person too
        ("Person", "you", "assistant", BIND_USER),
        ("Person", "I", "assistant", DROP),
        ("Person", "Frida", "user", KEEP),  # a named third party stays as extracted
    ],
)
def test_resolve_user_mentions(entity_type, surface, role, expected):
    mention = Mention(entity_type, surface, 0, len(surface), 0.9)
    assert resolve_user_mentions(mention, Segment(0, 10, role), Chunk("x", "h", user_id="u1")) == expected


def test_resolve_user_mentions_keeps_everything_when_the_chunk_has_no_user():
    mention = Mention("User", "I", 0, 1, 0.9)
    assert resolve_user_mentions(mention, Segment(0, 10, "user"), Chunk("x", "h")) == KEEP


# --- identity --------------------------------------------------------------------


def test_normalize_text_collapses_whitespace_and_case():
    assert _normalize_text("  Alice   JOHNSON ") == "alice johnson"


def test_entity_id_scopes():
    assert _entity_id("c1", "Location", "paris", "global", (0, 5)) == _entity_id(
        "c2", "Location", "paris", "global", (9, 14)
    )
    assert _entity_id("c1", "Product", "shoes", "chunk", (0, 5)) == _entity_id(
        "c1", "Product", "shoes", "chunk", (9, 14)
    )
    assert _entity_id("c1", "Product", "shoes", "chunk", (0, 5)) != _entity_id(
        "c2", "Product", "shoes", "chunk", (0, 5)
    )
    assert _entity_id("c1", "Quantity", "3", "span", (0, 1)) != _entity_id("c1", "Quantity", "3", "span", (5, 6))
    assert _entity_id("c1", "Location", "paris", "global", (0, 5)) == _entity_id(
        "c2", "Person", "paris", "global", (0, 5)
    )


# --- what gets written (real Memgraph) ---------------------------------------


def _user(memgraph, user_id="u1"):
    memgraph.query("MERGE (:User {user_id: $user_id})", params={"user_id": user_id})


@pytest.mark.asyncio
async def test_user_mentions_bind_to_the_users_node_with_the_turns_timestamp(memgraph):
    _user(memgraph)
    backend = _backend(
        surfaces={"I": "User", "Paris": "Location", "Kahlo": "User"},
        relations=[("visited", "I", "Paris", 0.8), ("visited", "Kahlo", "Paris", 0.7)],
    )
    document = _session(
        ("user", "I visited Paris", "2023-05-30T17:27:00+00:00"),
        ("assistant", "Kahlo visited Paris. I think so.", "2023-05-30T17:27:01+00:00"),
    )
    await from_documents([document], memgraph, backend)

    rows = memgraph.query(
        """
        MATCH (a)-[r:visited]->(b:gliner2)
        RETURN labels(a) AS head, coalesce(a.user_id, a.text) AS who, b.text AS place,
               toString(r.valid_at) AS valid_at, r.confidence AS confidence, r.chunk IS NOT NULL AS has_chunk
        ORDER BY who
        """
    )
    assert [(row["who"], row["place"], row["valid_at"]) for row in rows] == [
        ("Kahlo", "Paris", "2023-05-30T17:27:01.000000+00:00"),
        ("u1", "Paris", "2023-05-30T17:27:00.000000+00:00"),
    ]
    assert "User" in rows[1]["head"]
    assert all(row["has_chunk"] for row in rows)
    kahlo = memgraph.query("MATCH (n:gliner2 {text: 'Kahlo'}) RETURN n.entity_type AS entity_type")
    assert kahlo == [{"entity_type": "Person"}]  # re-typed, not a User node
    assert memgraph.query("MATCH (n:gliner2 {entity_type: 'User'}) RETURN count(n) AS n") == [{"n": 0}]
    assert backend.stats.mentions_bound_to_user == 1
    assert backend.stats.mentions_retyped == 1
    assert backend.stats.mentions_dropped == 1  # the assistant's "I"


@pytest.mark.asyncio
async def test_relations_never_cross_turns(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.8)])
    await from_documents(
        [_session(("user", "I am home", None), ("assistant", "Paris is nice", None))], memgraph, backend
    )
    assert memgraph.query("MATCH ()-[r:visited]->() RETURN count(r) AS n") == [{"n": 0}]


@pytest.mark.asyncio
async def test_global_identity_merges_across_chunks_and_links_every_mention(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"Paris": "Location", "shoes": "Product"})
    await from_documents(
        [
            _session(("user", "Paris and shoes", None)),
            _session(("user", "Paris again, other shoes", None)),
        ],
        memgraph,
        backend,
    )
    paris = memgraph.query(
        "MATCH (n:gliner2 {text: 'Paris'})-[:MENTIONED_IN]->(c:Chunk) RETURN count(DISTINCT n) AS nodes, count(c) AS chunks"
    )
    assert paris == [{"nodes": 1, "chunks": 2}]
    shoes = memgraph.query("MATCH (n:gliner2 {text: 'shoes'}) RETURN count(n) AS nodes")
    assert shoes == [{"nodes": 2}]  # chunk identity: one per session


@pytest.mark.asyncio
async def test_a_global_name_typed_differently_later_keeps_its_node_and_first_type(memgraph):
    """What a merge in a learned model relies on: relabelling a type never splits a name's node."""
    _user(memgraph)
    await from_documents(
        [_session(("user", "Paris was lovely", None))], memgraph, _backend(surfaces={"Paris": "Location"})
    )
    await from_documents(
        [_session(("user", "Paris called me", None))], memgraph, _backend(surfaces={"Paris": "Person"})
    )

    paris = memgraph.query(
        "MATCH (n:gliner2 {text: 'Paris'})-[:MENTIONED_IN]->(c:Chunk) "
        "RETURN collect(DISTINCT n.entity_type) AS types, count(DISTINCT n) AS nodes, count(c) AS chunks"
    )
    assert paris == [{"types": ["Location"], "nodes": 1, "chunks": 2}]


@pytest.mark.asyncio
async def test_a_lowercase_mention_of_a_global_type_stays_per_chunk(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"home": "Location", "Paris": "Location"})
    await from_documents(
        [_session(("user", "home and Paris", None)), _session(("user", "home again, Paris again", None))],
        memgraph,
        backend,
    )
    counts = {row["text"]: row["n"] for row in memgraph.query("MATCH (n:gliner2) RETURN n.text AS text, count(n) AS n")}
    assert counts == {"home": 2, "Paris": 1}


@pytest.mark.asyncio
async def test_span_identity_gives_every_value_mention_its_own_node(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "3": "Quantity"}, relations=[("owns_count", "I", "3", 0.9)])
    await from_documents([_session(("user", "I have 3 cats and 3 dogs", None))], memgraph, backend)
    assert memgraph.query("MATCH (n:gliner2 {text: '3'}) RETURN count(n) AS n") == [{"n": 2}]


@pytest.mark.asyncio
async def test_a_value_span_the_value_pass_claims_wins_over_the_main_pass(memgraph):
    """In a large vocabulary the main pass types a value as a domain type (`25:50` as Food, #386)."""
    _user(memgraph)
    backend = GLiNER2Backend(
        ontology=ONTOLOGY,
        model=FakeEngine(
            surfaces={"I": "User", "3 cats": "Product"},
            relations=[("visited", "I", "3 cats", 0.6)],
            value_surfaces={"I": "User", "3": "Quantity"},
            value_relations=[("owns_count", "I", "3", 0.9)],
        ),
    )
    await from_documents([_session(("user", "I have 3 cats", None))], memgraph, backend)

    rows = memgraph.query("MATCH (n:gliner2) RETURN n.entity_type AS type, n.text AS text")
    assert rows == [{"type": "Quantity", "text": "3"}]
    edges = memgraph.query("MATCH (:User)-[r]->(n:gliner2) RETURN type(r) AS type, n.text AS text")
    assert edges == [{"type": "owns_count", "text": "3"}]
    assert backend.stats.value_spans_claimed == 1


@pytest.mark.asyncio
async def test_a_value_relation_keeps_its_own_head_when_the_main_pass_types_that_span_otherwise(memgraph):
    """Aliased onto the main pass's Product, owns_count would leave its declared User/Person head."""
    _user(memgraph)
    backend = GLiNER2Backend(
        ontology=ONTOLOGY,
        model=FakeEngine(
            surfaces={"Paris": "Product"},
            value_surfaces={"Paris": "Person", "3": "Quantity"},
            value_relations=[("owns_count", "Paris", "3", 0.9)],
        ),
    )
    await from_documents([_session(("user", "Paris has 3 cats", None))], memgraph, backend)

    edges = memgraph.query("MATCH (h:gliner2)-[:owns_count]->(t:gliner2) RETURN h.entity_type AS head, t.text AS tail")
    assert edges == [{"head": "Person", "tail": "3"}]


@pytest.mark.asyncio
async def test_a_self_loop_left_by_identity_is_dropped_and_counted(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "my": "User"}, relations=[("mentions", "I", "my", 0.9)])
    await from_documents([_session(("user", "I love my life", None))], memgraph, backend)
    assert memgraph.query("MATCH (:User)-[r]->(:User) RETURN count(r) AS n") == [{"n": 0}]
    assert backend.stats.self_loops_dropped == 1


@pytest.mark.asyncio
async def test_the_same_fact_twice_in_a_chunk_is_one_edge_at_its_earliest_time(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.8)])
    document = _session(
        ("user", "I saw Paris", "2023-05-30T10:00:00+00:00"),
        ("user", "I loved Paris", "2023-05-30T09:00:00+00:00"),
    )
    await from_documents([document], memgraph, backend)
    rows = memgraph.query("MATCH ()-[r:visited]->() RETURN toString(r.valid_at) AS valid_at")
    assert rows == [{"valid_at": "2023-05-30T09:00:00.000000+00:00"}]


@pytest.mark.asyncio
async def test_the_same_fact_in_two_chunks_keeps_both_timestamps(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.8)])
    await from_documents(
        [
            _session(("user", "I saw Paris", "2023-01-01T00:00:00+00:00")),
            _session(("user", "I saw Paris again", "2023-06-01T00:00:00+00:00")),
        ],
        memgraph,
        backend,
    )
    assert memgraph.query("MATCH ()-[r:visited]->() RETURN count(r) AS n") == [{"n": 2}]


@pytest.mark.asyncio
async def test_reingesting_a_chunk_is_idempotent(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.8)])
    document = _session(("user", "I saw Paris", "2023-01-01T00:00:00+00:00"))
    await from_documents([document], memgraph, backend)
    await from_documents([document], memgraph, backend)
    assert memgraph.query("MATCH ()-[r:visited]->() RETURN count(r) AS n") == [{"n": 1}]
    assert memgraph.query("MATCH (n:gliner2) RETURN count(n) AS n") == [{"n": 1}]


def test_source_text_is_the_sentences_covering_both_spans():
    text = "user: Hello there. I flew to Paris on Monday. It rained!"
    head, tail = (text.index("I"), text.index("I") + 1), (text.index("Paris"), text.index("Paris") + 5)
    assert _source_text(text, (0, len(text)), head, tail) == "I flew to Paris on Monday."
    assert _source_text(text, (0, len(text)), (6, 11), tail) == "user: Hello there. I flew to Paris on Monday."


def test_source_text_is_capped_around_the_spans():
    text = "x " * 400 + "I flew to Paris" + " y" * 400
    head, tail = (text.index("I"), text.index("I") + 1), (text.index("Paris"), text.index("Paris") + 5)
    source = _source_text(text, (0, len(text)), head, tail)
    assert "I flew to Paris" in source
    assert len(source) <= 300


@pytest.mark.asyncio
async def test_mentions_record_which_turns_they_came_from(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"Paris": "Location", "Rome": "Location"})
    document = _session(
        ("user", "Paris first", None, "turn-1"),
        ("assistant", "Paris and Rome", None, "turn-2"),
    )
    await from_documents([document], memgraph, backend)
    rows = memgraph.query(
        "MATCH (n:gliner2)-[m:MENTIONED_IN]->(:Chunk) RETURN n.text AS text, m.sources AS sources ORDER BY text"
    )
    assert rows == [{"text": "Paris", "sources": ["turn-1", "turn-2"]}, {"text": "Rome", "sources": ["turn-2"]}]


@pytest.mark.asyncio
async def test_an_edge_carries_its_turn_speaker_and_sentence(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.8)])
    document = _session(("user", "Hi. I visited Paris last week. It was great.", "2023-05-30T17:27:00+00:00", "turn-1"))
    await from_documents([document], memgraph, backend)
    rows = memgraph.query("MATCH ()-[r:visited]->() RETURN r.source_id AS source_id, r.role AS role, r.text AS text")
    assert rows == [{"source_id": "turn-1", "role": "user", "text": "I visited Paris last week."}]


@pytest.mark.asyncio
async def test_the_same_fact_in_two_turns_is_two_edges_with_their_own_times(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.8)])
    document = _session(
        ("user", "I saw Paris", "2023-05-30T09:00:00+00:00", "turn-1"),
        ("user", "I loved Paris", "2023-05-30T10:00:00+00:00", "turn-2"),
    )
    await from_documents([document], memgraph, backend)
    rows = memgraph.query(
        "MATCH ()-[r:visited]->() RETURN r.source_id AS source_id, toString(r.valid_at) AS valid_at, r.text AS text "
        "ORDER BY source_id"
    )
    assert rows == [
        {"source_id": "turn-1", "valid_at": "2023-05-30T09:00:00.000000+00:00", "text": "user: I saw Paris"},
        {"source_id": "turn-2", "valid_at": "2023-05-30T10:00:00.000000+00:00", "text": "user: I loved Paris"},
    ]


@pytest.mark.asyncio
async def test_reingesting_a_chunk_with_turn_provenance_is_idempotent(memgraph):
    _user(memgraph)
    backend = _backend(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.8)])
    document = _session(
        ("user", "I saw Paris", "2023-05-30T09:00:00+00:00", "turn-1"),
        ("user", "I loved Paris", "2023-05-30T10:00:00+00:00", "turn-2"),
    )
    await from_documents([document], memgraph, backend)
    await from_documents([document], memgraph, backend)
    assert memgraph.query("MATCH ()-[r:visited]->() RETURN count(r) AS n") == [{"n": 2}]
    assert memgraph.query("MATCH (:gliner2)-[m:MENTIONED_IN]->() RETURN m.sources AS sources") == [
        {"sources": ["turn-1", "turn-2"]}
    ]


@pytest.mark.asyncio
async def test_an_infeasible_window_is_counted_as_a_bug(memgraph, caplog):
    backend = _backend(surfaces={"Paris": "Location"}, feasible=False)
    await from_documents([_session(("user", "Paris", None), user_id=None)], memgraph, backend)
    assert backend.stats.infeasible_windows == 1
    assert "infeasible" in caplog.text


@pytest.mark.asyncio
async def test_confidence_thresholds_filter_mentions_and_relations(memgraph):
    _user(memgraph)
    engine = FakeEngine(surfaces={"I": "User", "Paris": "Location"}, relations=[("visited", "I", "Paris", 0.4)])
    backend = GLiNER2Backend(ontology=ONTOLOGY, model=engine, relation_confidence_threshold=0.5)
    await from_documents([_session(("user", "I saw Paris", None))], memgraph, backend)
    assert memgraph.query("MATCH ()-[r:visited]->() RETURN count(r) AS n") == [{"n": 0}]

    memgraph.query("MATCH (n) DETACH DELETE n")
    backend = GLiNER2Backend(ontology=ONTOLOGY, model=engine, entity_confidence_threshold=0.95)
    await from_documents([_session(("user", "I saw Paris", None))], memgraph, backend)
    assert memgraph.query("MATCH (n:gliner2) RETURN count(n) AS n") == [{"n": 0}]
