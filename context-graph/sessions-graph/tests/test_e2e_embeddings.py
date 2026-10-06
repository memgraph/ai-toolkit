"""Embedding a session for recall, against a real Memgraph with MAGE.

The messages are recorded through a real ActionsGraph. The chunk, entity and
edge are written by hand in the shape unstructured2graph's extraction leaves
(``HAS_CHUNK``, ``MENTIONED_IN``, an edge carrying ``chunk`` and ``text``):
running a real extractor here would test extraction, not embedding.
"""

from __future__ import annotations

import pytest
from sessions_graph.cli import main
from sessions_graph.embeddings import DEFAULT_EMBEDDING_MODEL, EmbeddingUnavailableError, check_available
from sessions_graph.passages import split_passages

pytest.importorskip("actions_graph")

from actions_graph import MessageRole, Session

BGE_SMALL_DIMENSION = 384


@pytest.fixture(autouse=True)
def _requires_mage(memgraph):
    try:
        check_available(memgraph)
    except EmbeddingUnavailableError as exc:
        pytest.skip(f"Memgraph can't embed (needs MAGE): {exc}")


@pytest.fixture
def session(memgraph, actions_graph):
    """One session: two turns, a tool call, and one extracted entity and edge."""
    actions_graph.ensure_session(Session(session_id="s1", started_at="2026-10-05T10:00:00"))
    user = actions_graph.record_message(session_id="s1", role=MessageRole.USER, content="I adopted a beagle named Max")
    actions_graph.record_message(session_id="s1", role=MessageRole.ASSISTANT, content="Congratulations on Max!")
    actions_graph.record_tool_call(session_id="s1", tool_name="Read", tool_input={"file_path": "pets.md"})
    memgraph.query(
        """
        MATCH (a:Action {action_id: $turn})
        MERGE (u:User {user_id: 'alice'})
        CREATE (a)-[:HAS_CHUNK]->(c:Chunk {hash: 'h1', text: 'I adopted a beagle named Max'})
        CREATE (n:Entity {text: 'Max'})-[:MENTIONED_IN {sources: [$turn]}]->(c)
        CREATE (u)-[:adopted {chunk: 'h1', text: 'I adopted a beagle named Max', source_id: $turn}]->(n)
        """,
        {"turn": user.action_id},
    )
    return "s1"


def _vectors(memgraph):
    return memgraph.query(
        """
        MATCH (x:Action)
        RETURN x.action_type AS kind, size(x.passage_embeddings[0]) AS dims, x.embedding_model AS model
        UNION ALL
        MATCH (x:Entity) RETURN 'entity' AS kind, size(x.embedding) AS dims, x.embedding_model AS model
        UNION ALL
        MATCH ()-[x:adopted]->() RETURN 'edge' AS kind, size(x.embedding) AS dims, x.embedding_model AS model
        """
    )


def test_embeds_messages_entities_and_edges_but_not_tool_calls(graph, memgraph, session):
    embedded = graph.embed_session(session)

    assert (embedded.messages, embedded.entities, embedded.edges) == (2, 1, 1)
    by_kind = {row["kind"]: row for row in _vectors(memgraph)}
    for kind in ("user_message", "assistant_message", "entity", "edge"):
        assert by_kind[kind]["dims"] == BGE_SMALL_DIMENSION
        assert by_kind[kind]["model"] == DEFAULT_EMBEDDING_MODEL
    assert by_kind["tool_call"]["dims"] is None

    status = memgraph.query(
        "MATCH (s:Session {session_id: 's1'}) RETURN s.embedding_status AS status, s.embedding_model AS model"
    )[0]
    assert status == {"status": "completed", "model": DEFAULT_EMBEDDING_MODEL}
    assert graph.get_pending_embedding_sessions() == []


def test_a_long_message_gets_one_vector_per_passage(graph, memgraph, actions_graph):
    actions_graph.ensure_session(Session(session_id="s2", started_at="2026-10-05T10:00:00"))
    text = "A sentence about the plan. " * 120
    actions_graph.record_message(session_id="s2", role=MessageRole.ASSISTANT, content=text)

    graph.embed_session("s2")

    rows = memgraph.query("MATCH (a:Action) RETURN size(a.passage_embeddings) AS n, a.passage_embeddings[1] AS v")
    assert rows[0]["n"] == len(split_passages(text)) > 1
    assert len(rows[0]["v"]) == BGE_SMALL_DIMENSION


def test_embedding_twice_embeds_nothing_new(graph, session):
    graph.embed_session(session)

    again = graph.embed_session(session)

    assert (again.messages, again.entities, again.edges) == (0, 0, 0)


def test_a_vector_from_another_model_is_replaced_not_kept_beside(graph, memgraph, session):
    """A model change must re-embed: recall comparing vectors from two models gets nonsense, not an error."""
    graph.embed_session(session)
    memgraph.query("MATCH (a:Action {action_type: 'user_message'}) SET a.embedding_model = 'an-older-model'")
    memgraph.query("MATCH (s:Session {session_id: 's1'}) SET s.embedding_model = 'an-older-model'")

    assert graph.get_pending_embedding_sessions() == ["s1"]
    again = graph.embed_session(session)

    assert (again.messages, again.entities, again.edges) == (1, 0, 0)
    models = {row["model"] for row in _vectors(memgraph) if row["dims"]}
    assert models == {DEFAULT_EMBEDDING_MODEL}


def test_a_model_that_cannot_load_is_recorded_on_the_session(graph, memgraph, session):
    with pytest.raises(EmbeddingUnavailableError):
        graph.embed_session(session, model="no-such-org/no-such-model")

    row = memgraph.query(
        "MATCH (s:Session {session_id: 's1'}) RETURN s.embedding_status AS status, s.embedding_error AS error"
    )[0]
    assert row["status"] == "failed"
    assert "no-such-org/no-such-model" in row["error"]
    assert graph.get_pending_embedding_sessions() == ["s1"]


def test_cli_embeds_pending_sessions(graph, session, capsys):
    assert main(["embed", "--pending", "--model", DEFAULT_EMBEDDING_MODEL]) == 0

    assert "OK s1: 2 messages, 1 entities, 1 edges" in capsys.readouterr().out
    assert graph.get_pending_embedding_sessions() == []


def test_cli_reports_a_failure_and_exits_nonzero(session, capsys):
    assert main(["embed", "--session", session, "--model", "no-such-org/no-such-model"]) == 1

    assert "FAILED s1" in capsys.readouterr().err
