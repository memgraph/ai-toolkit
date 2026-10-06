"""Recall against a real Memgraph with MAGE: real messages, real text index, real vectors.

The chunks, entities and edges are written by hand in the shape extraction
leaves (``HAS_CHUNK``, ``MENTIONED_IN`` with its source turns, an edge with
``chunk``, ``source_id``, ``role`` and ``text``): running a real extractor
here would test extraction, not recall.
"""

from __future__ import annotations

import pytest
from sessions_graph import RecallConfig
from sessions_graph.embeddings import EmbeddingUnavailableError, check_available

pytest.importorskip("actions_graph")

from actions_graph import MessageRole, Session


@pytest.fixture(autouse=True)
def _requires_mage(memgraph):
    try:
        check_available(memgraph)
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"Memgraph can't embed (needs MAGE): {exc}")


def _session(actions_graph, memgraph, *, user, session, when, said, reply=None):
    """One session of ``user``: their message (and an optional reply), returning the message's action id."""
    actions_graph.ensure_session(Session(session_id=session, started_at=when))
    memgraph.query(
        "MERGE (u:User {user_id: $u}) WITH u MATCH (s:Session {session_id: $s}) MERGE (u)-[:HAD_SESSION]->(s)",
        {"u": user, "s": session},
    )
    turn = actions_graph.record_message(session_id=session, role=MessageRole.USER, content=said, timestamp=when)
    if reply:
        actions_graph.record_message(session_id=session, role=MessageRole.ASSISTANT, content=reply, timestamp=when)
    return turn.action_id


def _fact(memgraph, *, user, turn, relation, entity, sentence, valid_at="2023-05-30T17:27:00+00:00"):
    """An extracted edge from ``user`` to ``entity``, read from ``turn``, through that turn's chunk."""
    memgraph.query(
        "MATCH (a:Action {action_id: $turn}), (u:User {user_id: $user}) "
        "MERGE (c:Chunk {hash: $turn}) MERGE (a)-[:HAS_CHUNK]->(c) "
        "MERGE (n:Entity {text: $entity}) "
        "MERGE (n)-[m:MENTIONED_IN]->(c) SET m.sources = [$turn] "
        f"CREATE (u)-[:{relation} {{chunk: $turn, source_id: $turn, role: 'user', confidence: 0.9, "
        "valid_at: datetime($valid_at), text: $sentence}]->(n)",
        {"turn": turn, "user": user, "entity": entity, "sentence": sentence, "valid_at": valid_at},
    )


def _ready(graph, memgraph):
    graph.setup()
    for row in memgraph.query("MATCH (s:Session) RETURN s.session_id AS id"):
        graph.embed_session(row["id"])


@pytest.fixture
def moma(graph, memgraph, actions_graph):
    turn = _session(
        actions_graph,
        memgraph,
        user="u1",
        session="s1",
        when="2023-05-30T17:27:00+00:00",
        said="I went to the Museum of Modern Art. It was wonderful.",
        reply="Glad you enjoyed it!",
    )
    _fact(memgraph, user="u1", turn=turn, relation="visited", entity="Museum of Modern Art",
          sentence="I went to the Museum of Modern Art.")  # fmt: skip
    _ready(graph, memgraph)


def test_the_facts_lane_reaches_the_turn_a_fact_was_read_from(graph, moma):
    recalled = graph.recall("u1", "When did I visit the Museum of Modern Art?", config=RecallConfig(lanes=("facts",)))

    lines = recalled.lines()
    assert (
        'FACT: user -[visited @ 2023-05-30]-> Museum of Modern Art -- user: "I went to the Museum of Modern Art."'
        in lines
    )
    assert any(line.startswith("TURN [session s1, 2023-05-30T17:27, user]") for line in lines)


def test_without_the_graph_lanes_only_turns_come_back(graph, moma):
    recalled = graph.recall("u1", "Museum of Modern Art", config=RecallConfig(lanes=("turns", "text")))

    assert recalled.turns
    assert not recalled.facts


def test_user_facts_gathers_every_fact_of_a_relation_type_across_sessions(graph, memgraph, actions_graph):
    """The aggregation a counting question needs: every wedding the user attended, whichever session."""
    for i, couple in enumerate(("Rachel and Mike", "Emily and Sarah", "Jen and Tom")):
        turn = _session(actions_graph, memgraph, user="u1", session=f"s{i}", when=f"2023-0{i + 1}-01T10:00:00+00:00",
                        said=f"I attended the wedding of {couple}.")  # fmt: skip
        _fact(memgraph, user="u1", turn=turn, relation="attended", entity=f"wedding of {couple}",
              sentence=f"I attended the wedding of {couple}.")  # fmt: skip
    _fact(memgraph, user="u1", turn=turn, relation="likes", entity="cake", sentence="I like cake.")
    _ready(graph, memgraph)

    recalled = graph.recall(
        "u1", "How many weddings have I attended?", config=RecallConfig(lanes=("user_facts",), user_fact_types=1)
    )

    assert [fact.type for fact in recalled.facts] == ["attended"] * 3


def test_every_lane_reads_only_the_asking_users_history(graph, memgraph, actions_graph):
    """Two people who both visited Paris: one shared Paris entity, each with their own session and fact."""
    for user, companion in (("u1", "Anna"), ("u2", "Bob")):
        turn = _session(actions_graph, memgraph, user=user, session=f"{user}-s", when="2023-05-30T17:27:00+00:00",
                        said=f"I visited Paris with {companion}.")  # fmt: skip
        _fact(memgraph, user=user, turn=turn, relation="visited", entity="Paris",
              sentence=f"I visited Paris with {companion}.")  # fmt: skip
    _ready(graph, memgraph)

    for user, own, other in (("u1", "Anna", "Bob"), ("u2", "Bob", "Anna")):
        text = "\n".join(graph.recall(user, "Who did I visit Paris with?").lines())
        assert own in text
        assert other not in text


def test_turns_come_before_facts_and_both_in_time_order(graph, memgraph, actions_graph, moma):
    """Turns are the answer store: after ~45 fact rows their evidence was read past.
    Time order lets "which came first" and "what is current" read off the sequence."""
    _session(actions_graph, memgraph, user="u1", session="s0", when="2023-01-02T09:00:00+00:00",
             said="I went to the Museum of Modern Art with Anna.")  # fmt: skip
    _ready(graph, memgraph)

    lines = graph.recall("u1", "Museum of Modern Art", config=RecallConfig(lanes=("turns", "facts"))).lines()

    kinds = [line.split(" ", 1)[0] for line in lines]
    assert kinds == sorted(kinds, key=lambda kind: kind != "TURN")
    turns = [line for line in lines if line.startswith("TURN")]
    assert turns[0].startswith("TURN [session s0, 2023-01-02T09:00")
    assert turns == sorted(turns, key=lambda line: line.split(", ")[1])


def test_when_memgraph_cannot_embed_text_search_answers_and_says_so(graph, moma):
    recalled = graph.recall("u1", "Museum of Modern Art", model="no-such-org/no-such-model")

    assert recalled.vector_lanes_off
    assert any("Museum of Modern Art" in turn.text for turn in recalled.turns)
    assert "Vector search is off" in recalled.render()


def test_render_puts_the_reading_rules_and_date_before_the_rows(graph, moma):
    recalled = graph.recall("u1", "When did I visit the Museum of Modern Art?")

    rendered = recalled.render(today="2023-06-01")

    head, rows = rendered.split("\n\nRows:\n")
    assert "Today is 2023-06-01." in head
    assert "most recent one is current" in head
    assert rows.splitlines() == recalled.lines()
    assert recalled.to_json()["turns"][0]["session_id"] == "s1"


def test_an_empty_memory_says_so(graph, memgraph):
    graph.setup()
    memgraph.query("CREATE (:User {user_id: 'nobody'})")

    recalled = graph.recall("nobody", "What did I eat yesterday?")

    assert recalled.lines() == []
    assert "(nothing in memory matched)" in recalled.render()


def test_config_overrides_widths_and_lanes_and_rejects_nonsense():
    config = RecallConfig.from_mapping({"turns_k": "4", "lanes": "turns,text", "unrelated": "x"})

    assert (config.turns_k, config.lanes, config.facts_k) == (4, ("turns", "text"), RecallConfig().facts_k)
    with pytest.raises(ValueError, match="unknown recall lanes"):
        RecallConfig.from_mapping({"lanes": "turns,telepathy"})
    with pytest.raises(ValueError, match=">= 0"):
        RecallConfig.from_mapping({"turns_k": -1})


def _entity_fact(memgraph, *, turn, relation, head, tail, sentence):
    """An extracted edge between two entities, read from ``turn``: a learned domain fact not headed by the User."""
    memgraph.query(
        "MATCH (a:Action {action_id: $turn}) "
        "MERGE (c:Chunk {hash: $turn}) MERGE (a)-[:HAS_CHUNK]->(c) "
        "MERGE (h:Entity {text: $head}) MERGE (t:Entity {text: $tail}) "
        "MERGE (h)-[m:MENTIONED_IN]->(c) SET m.sources = [$turn] "
        "MERGE (t)-[n:MENTIONED_IN]->(c) SET n.sources = [$turn] "
        f"CREATE (h)-[:{relation} {{chunk: $turn, source_id: $turn, role: 'user', confidence: 0.9, "
        "valid_at: datetime('2023-05-30T17:27:00+00:00'), text: $sentence}]->(t)",
        {"turn": turn, "head": head, "tail": tail, "sentence": sentence},
    )


def test_user_facts_reads_facts_whose_head_is_not_the_user(graph, memgraph, actions_graph):
    """A learned relation like `depends_on` hangs off a project, not the user (#439); another user's stays out."""
    for i, (user, project, library) in enumerate(
        (("u1", "ingest", "tokio"), ("u1", "billing", "serde"), ("u2", "crawler", "reqwest"))
    ):
        turn = _session(actions_graph, memgraph, user=user, session=f"s{i}", when=f"2023-0{i + 1}-01T10:00:00+00:00",
                        said=f"Our {project} service depends on {library}.")  # fmt: skip
        _entity_fact(memgraph, turn=turn, relation="depends_on", head=project, tail=library,
                     sentence=f"Our {project} service depends on {library}.")  # fmt: skip
    _ready(graph, memgraph)

    recalled = graph.recall(
        "u1", "What do my services depend on?", config=RecallConfig(lanes=("user_facts",), user_fact_types=1)
    )

    assert sorted((fact.head, fact.tail) for fact in recalled.facts) == [("billing", "serde"), ("ingest", "tokio")]


def test_user_fact_types_rank_by_their_description_from_the_adopted_version(graph, memgraph, actions_graph, tmp_path):
    """`pins` says nothing on its own; its description is what a question about locked versions matches."""
    turn = _session(actions_graph, memgraph, user="u1", session="s1", when="2023-01-01T10:00:00+00:00",
                    said="I pinned serde to 1.0.188 and use tokio everywhere.")  # fmt: skip
    _entity_fact(memgraph, turn=turn, relation="pins", head="ingest", tail="serde 1.0.188",
                 sentence="I pinned serde to 1.0.188.")  # fmt: skip
    _entity_fact(memgraph, turn=turn, relation="uses_library", head="ingest", tail="tokio",
                 sentence="I use tokio everywhere.")  # fmt: skip
    schema = tmp_path / "coding.yaml"
    schema.write_text(
        "entity_types:\n"
        "  - {label: User, description: the user, identity: global}\n"
        "  - {label: Person, description: someone else, identity: global}\n"
        "  - {label: Project, description: a code project, identity: global}\n"
        "  - {label: Library, description: a code library, identity: global}\n"
        "relation_types:\n"
        "  - {label: pins, description: 'locks a dependency to one exact release number', "
        "start_labels: [Project], end_labels: [Library]}\n"
        "  - {label: uses_library, description: 'calls into a package at runtime', "
        "start_labels: [Project], end_labels: [Library]}\n"
    )
    graph.supply_ontology_file("u1", schema, derive="off")
    _ready(graph, memgraph)

    recalled = graph.recall(
        "u1", "Which exact release did I lock it to?", config=RecallConfig(lanes=("user_facts",), user_fact_types=1)
    )

    assert [fact.type for fact in recalled.facts] == ["pins"]


def test_a_user_without_a_version_ranks_types_by_the_default_models_descriptions(memgraph):
    from sessions_graph.recall import _type_descriptions

    descriptions = _type_descriptions(memgraph, "nobody")

    assert descriptions["works_for"] == "is employed by or works at"
