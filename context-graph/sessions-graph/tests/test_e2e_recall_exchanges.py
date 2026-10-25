"""Exchange recall on real Memgraph, with fixed vectors and no model or judge calls."""

from __future__ import annotations

import importlib
from dataclasses import replace

import pytest
import tiktoken
from sessions_graph import RecallConfig
from sessions_graph.recall import MemoryNote, Recalled
from sessions_graph.recall_exchanges import _NEIGHBORS, _discover, assemble

r = importlib.import_module("sessions_graph.recall")


@pytest.fixture
def episodes(memgraph, monkeypatch):
    """Owned, foreign, nested-agent and unselected-session episodes share cue terms."""
    memgraph.query(
        "CREATE (u:User {user_id: 'own'}), (v:User {user_id: 'foreign'}), "
        "(s:Session {session_id: 'selected'}), (t:Session {session_id: 'foreign'}), "
        "(n:Session {session_id: 'unselected'}), (agent:Agent), "
        "(u)-[:HAD_SESSION]->(s), (u)-[:HAD_SESSION]->(n), (v)-[:HAD_SESSION]->(t), "
        "(s)-[:HAS_AGENT]->(agent)"
    )
    documents = [
        ("a", "selected", "user_message", "How did we fix the widget?", [1.0, 0.0]),
        ("b", "selected", "assistant_message", "We fixed the widget by replacing its cable.", [0.9, 0.1]),
        ("c", "selected", "user_message", "Another issue was a broken socket.", [0.5, 0.5]),
        ("d", "selected", "assistant_message", "The socket problem was resolved.", [0.5, 0.5]),
        ("secret", "foreign", "assistant_message", "SECRET widget problem fixed", [1.0, 0.0]),
        ("unused", "unselected", "assistant_message", "UNSELECTED widget issue fixed", [0.0, 1.0]),
    ]
    for identifier, session, kind, text, vector in documents:
        memgraph.query(
            "MATCH (s:Session {session_id: $session}) "
            "CREATE (a:Action {action_id: $id, action_type: $kind, text: $text, timestamp: '2026-01-01', "
            "embedding_model: 'test', passage_embeddings: [$vector]}) CREATE (s)-[:HAS_ACTION]->(a)",
            {"id": identifier, "session": session, "kind": kind, "text": text, "vector": vector},
        )
    memgraph.query(
        "MATCH (a:Action {action_id: 'a'}), (b:Action {action_id: 'b'}), "
        "(c:Action {action_id: 'c'}), (d:Action {action_id: 'd'}), (x:Action {action_id: 'secret'}) "
        "CREATE (a)-[:FOLLOWED_BY]->(b)-[:FOLLOWED_BY]->(c)-[:FOLLOWED_BY]->(d), "
        "(a)-[:FOLLOWED_BY]->(x)"
    )
    # The embedding boundary is fixed; every retrieval query executes on Memgraph.
    monkeypatch.setattr(r, "embed_texts", lambda *_: [[1.0, 0.0]])
    return memgraph


def seed(db):
    """Only the first user turn is a seed; graph traversal must recover its response."""
    return r._assemble(db, "What problems did we fix?", ["a"], [], RecallConfig())


def test_neighbors_preserve_direction_session_and_episode_boundaries(episodes):
    pairs = episodes.query(_NEIGHBORS, {"user": "own", "ids": ["a", "b", "c", "d", "secret"]})
    assert {(p["source"], p["neighbor"]) for p in pairs} == {("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")}


def test_nested_agent_neighbors_stay_in_their_container(episodes):
    episodes.query(
        "MATCH (agent:Agent), (s:Session {session_id: 'selected'}) "
        "CREATE (a:Action {action_id: 'nested-user', action_type: 'user_message', text: 'nested issue'}), "
        "(b:Action {action_id: 'nested-answer', action_type: 'assistant_message', text: 'nested fix'}), "
        "(agent)-[:HAS_ACTION]->(a), (agent)-[:HAS_ACTION]->(b), (a)-[:FOLLOWED_BY]->(b) "
        "WITH a MATCH (outside:Action {action_id: 'b'}) CREATE (a)-[:FOLLOWED_BY]->(outside)"
    )
    assert episodes.query(_NEIGHBORS, {"user": "own", "ids": ["nested-user"]}) == [
        {"source": "nested-user", "neighbor": "nested-answer"}
    ]


def test_exchange_sdk_render_json_budget_and_sources(episodes):
    initial = seed(episodes)
    result = assemble(episodes, "own", initial, {"turns": ["a"], "text": []}, RecallConfig(), None)
    assert isinstance(result, Recalled)
    assert result.exchanges == (("a", "b"),)
    assert {turn.action_id for turn in result.turns} == {"a", "b"}
    assert not result.facts
    assert result.to_json()["exchanges"] == [["a", "b"]]
    assert "EXCHANGE" in result.render(today="2026-10-24")
    assert "cable" in result.render()
    assert sum(len(tiktoken.get_encoding("cl100k_base").encode(line)) + 1 for line in result.lines()) <= 6000
    empty = assemble(episodes, "own", initial, {"turns": ["a"]}, RecallConfig(token_budget=0), None)
    assert empty.lines() == [] and empty.turns == []


@pytest.mark.parametrize("count", [None, 10001])
def test_bounded_cues_exclude_other_sessions_and_preserve_full_sources(episodes, count):
    long_text = "issue fixed " * 1100 + " ORIGINAL SOURCE END"
    episodes.query("MATCH (a:Action {action_id: 'd'}) SET a.text = $text", {"text": long_text})
    initial = seed(episodes)
    turns = {turn.action_id: turn for turn in initial.turns}
    ids = _discover(episodes, "own", initial.question, turns, ["a"], count)
    assert ids and set(ids) <= {"a", "b", "c", "d"} and "d" in ids
    result = assemble(
        episodes,
        "own",
        initial,
        {"turns": ["a"], "text": []},
        RecallConfig(supplemental_search=True, turn_chars=20000, token_budget=20000),
        count,
    )
    assert next(turn.text for turn in result.turns if turn.action_id == "d") == long_text
    assert "SECRET" not in result.render() and "UNSELECTED" not in result.render()


def test_local_document_cap_preserves_primary_context(episodes):
    initial = seed(episodes)
    plain = assemble(episodes, "own", initial, {"turns": ["a"], "text": []}, RecallConfig(), None)
    episodes.query(
        "MATCH (s:Session {session_id: 'selected'}) UNWIND range(1, 513) AS i "
        "CREATE (a:Action {action_id: 'noise-' + toString(i), text: 'problem fixed'}) CREATE (s)-[:HAS_ACTION]->(a)"
    )
    capped = assemble(
        episodes, "own", initial, {"turns": ["a"], "text": []}, RecallConfig(supplemental_search=True), None
    )
    assert capped == plain


@pytest.mark.parametrize("lanes", [("turns",), ("text",), ()])
def test_disabled_lanes_and_text_fallback_return_normal_sdk(episodes, lanes):
    config = RecallConfig(lanes=lanes, packing="exchanges", supplemental_search=True)
    # Empty text query avoids unrelated index setup in this lane/fallback check.
    question = "the" if "text" in lanes else "widget problems"
    result = r.recall(episodes, "own", question, config=config, model="test")
    assert isinstance(result, Recalled)
    assert "SECRET" not in result.render()
    assert not result.facts
    if not lanes or lanes == ("text",):
        assert result.lines() == []

    def unavailable(*_):
        raise r.EmbeddingUnavailableError("offline")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(r, "embed_texts", unavailable)
        fallback = r.recall(episodes, "own", "the", config=config, model="test")
    assert isinstance(fallback, Recalled) and fallback.vector_lanes_off == "offline"
    assert fallback.lines() == []


def test_memory_priority_and_special_tokens_fit_budget(episodes):
    initial = replace(seed(episodes), memories=[MemoryNote("notes", "2026-01-01", "<|endoftext|> saved note")])
    config = RecallConfig(token_budget=30)
    result = assemble(episodes, "own", initial, {"turns": ["a"]}, config, None)
    assert result.memories == initial.memories
    assert result.turns == []


def test_mapping_validates_options_and_preserves_none():
    config = RecallConfig.from_mapping({"packing": "exchanges", "supplemental_search": True, "query_instruction": None})
    assert config.packing == "exchanges" and config.supplemental_search and config.query_instruction is None
    for values in ({"packing": "other"}, {"supplemental_search": "sometimes"}, {"token_budget": -1}):
        with pytest.raises(ValueError):
            RecallConfig.from_mapping(values)


def test_shared_action_does_not_grant_access_to_foreign_container(episodes):
    episodes.query(
        "MATCH (t:Session {session_id: 'foreign'}), (a:Action {action_id: 'a'}) CREATE (t)-[:HAS_ACTION]->(a)"
    )
    assert episodes.query(_NEIGHBORS, {"user": "own", "ids": ["a"]}) == [{"source": "a", "neighbor": "b"}]


def test_global_and_bounded_discovery_select_owned_episodes(episodes):
    from sessions_graph import SessionsGraph

    # Use the real setup path to create a text index; no source extraction.
    SessionsGraph(episodes).setup()
    initial = seed(episodes)
    turns = {turn.action_id: turn for turn in initial.turns}
    for count in (6, 10000, 10001):
        ids = _discover(episodes, "own", initial.question, turns, ["a"], count)
        assert ids and set(ids) <= {"a", "b", "c", "d"}
    result = r.recall(
        episodes,
        "own",
        initial.question,
        model="test",
        config=RecallConfig(lanes=("turns", "text"), packing="exchanges", supplemental_search=True),
    )
    assert isinstance(result, Recalled) and result.exchanges
    assert "SECRET" not in result.render()


def test_nonempty_text_fallback_preserves_exchanges_and_ownership(episodes, monkeypatch):
    from sessions_graph import SessionsGraph

    SessionsGraph(episodes).setup()

    def unavailable(*_):
        raise r.EmbeddingUnavailableError("offline")

    monkeypatch.setattr(r, "embed_texts", unavailable)
    result = r.recall(
        episodes,
        "own",
        "widget",
        model="test",
        config=RecallConfig(lanes=("turns", "text"), packing="exchanges", supplemental_search=True),
    )
    assert result.vector_lanes_off == "offline" and result.exchanges
    assert {"a", "b"} <= {turn.action_id for turn in result.turns} <= {"a", "b", "c", "d", "unused"}
    assert "cable" in result.render() and "SECRET" not in result.render()
    assert ["a", "b"] in result.to_json()["exchanges"]
