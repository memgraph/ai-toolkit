"""Recall queries on real Memgraph, with hand-written vectors and no model calls."""

from __future__ import annotations

import importlib

import pytest
from sessions_graph import RecallConfig

recall = importlib.import_module("sessions_graph.recall")


@pytest.fixture
def shared_chunk(memgraph):
    """Several owned turns and a foreign turn share one content-addressed chunk."""
    memgraph.query(
        "CREATE (u:User {user_id: 'own'}), (v:User {user_id: 'other'}), "
        "(s:Session {session_id: 'own'}), (t:Session {session_id: 'other'}), "
        "(u)-[:HAD_SESSION]->(s), (v)-[:HAD_SESSION]->(t), "
        "(a:Action {action_id: 'z-first', text: 'own first', embedding_model: 'test', "
        "passage_embeddings: [[1.0, 0.0], [1.0, 0.0]]}), "
        "(b:Action {action_id: 'a-second', text: 'own second', embedding_model: 'test', "
        "passage_embeddings: [[1.0, 0.0], [1.0, 0.0]]}), "
        "(d:Action {action_id: 'nested', text: 'own nested', embedding_model: 'test', "
        "passage_embeddings: [[0.0, 1.0]]}), "
        "(x:Action {action_id: 'foreign', text: 'foreign', embedding_model: 'test', "
        "passage_embeddings: [[1.0, 0.0]]}), (agent:Agent), "
        "(s)-[:HAS_ACTION]->(a), (s)-[:HAS_ACTION]->(b), "
        "(s)-[:HAS_AGENT]->(agent)-[:HAS_ACTION]->(d), (s)-[:HAS_ACTION]->(d), "
        "(t)-[:HAS_ACTION]->(x), (c:Chunk {hash: 'shared'}), "
        "(a)-[:HAS_CHUNK]->(c), (b)-[:HAS_CHUNK]->(c), (d)-[:HAS_CHUNK]->(c), (x)-[:HAS_CHUNK]->(c), "
        "(n:Entity {text: 'own', embedding_model: 'test', embedding: [1.0, 0.0]}), "
        "(m:Entity {text: 'shared', embedding_model: 'test', embedding: [1.0, 0.0]}), "
        "(f:Entity {text: 'foreign', embedding_model: 'test', embedding: [1.0, 0.0]}), "
        "(z:Entity {text: 'unattributed', embedding_model: 'test', embedding: [1.0, 0.0]}), "
        "(e:Entity {text: 'empty sources', embedding_model: 'test', embedding: [1.0, 0.0]}), "
        "(n)-[:MENTIONED_IN {sources: ['z-first']}]->(c), "
        "(m)-[:MENTIONED_IN {sources: ['foreign', 'a-second', 'nested']}]->(c), "
        "(f)-[:MENTIONED_IN {sources: ['foreign']}]->(c), "
        "(z)-[:MENTIONED_IN]->(c), (e)-[:MENTIONED_IN {sources: []}]->(c), "
        "(n)-[:owns {source_id: 'z-first', text: 'first fact', confidence: 0.9, "
        "embedding_model: 'test', embedding: [1.0, 0.0]}]->(m), "
        "(m)-[:owns {source_id: 'a-second', text: 'second fact', confidence: 0.9, "
        "embedding_model: 'test', embedding: [1.0, 0.0]}]->(n), "
        "(m)-[:owns {source_id: 'nested', text: 'nested fact', confidence: 0.9, "
        "embedding_model: 'test', embedding: [1.0, 0.0]}]->(n), "
        "(f)-[:private {source_id: 'foreign', text: 'foreign fact', confidence: 1.0, "
        "embedding_model: 'test', embedding: [1.0, 0.0]}]->(m)"
    )
    return {"user": "own", "query": [1.0, 0.0], "model": "test", "k": 50, "types": ["owns", "private"]}


def test_shared_chunks_keep_entity_mentions_and_fact_sources_owned(memgraph, shared_chunk):
    expected_entities = memgraph.query(
        "MATCH (n:Entity) WHERE n.text IN ['own', 'shared'] RETURN id(n) AS id ORDER BY id"
    )
    assert memgraph.query(recall._ENTITIES, shared_chunk) == expected_entities
    assert memgraph.query(recall._ENTITIES, {**shared_chunk, "k": 1}) == expected_entities[:1]

    expected_sources = [
        row["source"] for row in memgraph.query("MATCH ()-[r:owns]->() RETURN r.source_id AS source ORDER BY id(r)")
    ]
    for query in (recall._FACTS, recall._USER_FACTS):
        facts = memgraph.query(query, shared_chunk)
        assert [row["turn"] for row in facts] == expected_sources
        assert {row["sentence"] for row in facts} == {"first fact", "second fact", "nested fact"}
        assert memgraph.query(query, {**shared_chunk, "k": 1}) == facts[:1]
    assert memgraph.query(recall._USER_FACT_TYPES, shared_chunk) == [{"type": "owns"}]

    entity_facts = memgraph.query(
        recall._ENTITY_FACTS,
        {"user": "own", "ids": [row["id"] for row in expected_entities], "per": 10},
    )
    assert {row["turn"] for row in entity_facts} == set(expected_sources)
    assert len(entity_facts) == 3


def test_tied_turn_passages_use_action_id_then_passage(memgraph, shared_chunk):
    assert memgraph.query(recall._TURNS, shared_chunk) == [
        {"id": "a-second", "passage": 0},
        {"id": "a-second", "passage": 1},
        {"id": "z-first", "passage": 0},
        {"id": "z-first", "passage": 1},
        {"id": "nested", "passage": 0},
    ]
    assert memgraph.query(recall._TURNS, {**shared_chunk, "k": 1}) == [{"id": "a-second", "passage": 0}]


@pytest.mark.parametrize(
    ("k", "expected"),
    [
        (0, []),
        (2, [("a-second", 0), ("a-second", 1), ("z-first", 0)]),
        (3, [("a-second", 0), ("a-second", 1), ("z-first", 0), ("nested", 0)]),
    ],
)
def test_turn_diversity_preserves_original_passages_without_adding_to_them(memgraph, shared_chunk, k, expected):
    rows = memgraph.query(recall._TURNS, {**shared_chunk, "k": k})
    assert [(row["id"], row["passage"]) for row in rows] == expected


def test_turn_diversity_handles_missing_users_models_and_empty_vectors(memgraph, shared_chunk):
    memgraph.query(
        "MATCH (s:Session {session_id: 'own'}) CREATE (s)-[:HAS_ACTION]->"
        "(:Action {action_id: 'empty', text: 'empty embedding', embedding_model: 'test', passage_embeddings: []})"
    )
    assert memgraph.query(recall._TURNS, {**shared_chunk, "user": "absent"}) == []
    assert memgraph.query(recall._TURNS, {**shared_chunk, "model": "absent"}) == []
    assert len(memgraph.query(recall._TURNS, shared_chunk)) == 5


def test_turn_row_batch_preserves_nested_paths_and_deduplicates_requested_ids(memgraph, shared_chunk):
    params = {"ids": ["z-first", "nested", "a-second", "nested", "absent"]}
    rows = memgraph.query(recall._TURN_ROWS, params)
    assert len(rows) == 3
    assert {row["id"] for row in rows} == {"z-first", "nested", "a-second"}
    assert {row["session"] for row in rows} == {"own"}
    assert [turn.action_id for turn in recall._turns(memgraph, params["ids"])] == ["a-second", "nested", "z-first"]
    assert memgraph.query(recall._TURN_ROWS, {"ids": []}) == []


def test_entity_fact_source_batch_keeps_owned_actions_without_text(memgraph, shared_chunk):
    memgraph.query(
        "MATCH (s:Session {session_id: 'own'}), (n:Entity {text: 'own'}), (m:Entity {text: 'shared'}) "
        "CREATE (s)-[:HAS_ACTION]->(:Action {action_id: 'metadata-only'}), "
        "(n)-[:owns {source_id: 'metadata-only', text: 'metadata fact', confidence: 1.0}]->(m)"
    )
    ids = [row["id"] for row in memgraph.query(recall._ENTITIES, shared_chunk)]
    rows = memgraph.query(recall._ENTITY_FACTS, {"user": "own", "ids": ids, "per": 20})
    assert {row["turn"] for row in rows} == {"z-first", "a-second", "nested", "metadata-only"}
    assert len(rows) == 4
    assert memgraph.query(recall._ENTITY_FACTS, {"user": "absent", "ids": ids, "per": 20}) == []


def test_entity_fact_source_ownership_precedes_confidence_cutoff(memgraph, shared_chunk):
    entity = memgraph.query(
        "MATCH (n:Entity {text: 'shared'}), (m:Entity {text: 'foreign'}) "
        "CREATE (n)-[:private {source_id: 'orphan', text: 'orphan fact', confidence: 1.0}]->(m) "
        "RETURN id(n) AS id"
    )[0]["id"]
    params = {"user": "own", "ids": [entity], "per": 1}
    assert [row["turn"] for row in memgraph.query(recall._ENTITY_FACTS, params)] == ["z-first"]
    assert memgraph.query(recall._ENTITY_FACTS, {**params, "ids": []}) == []
    assert memgraph.query(recall._ENTITY_FACTS, {**params, "per": 0}) == []


def test_entity_source_must_belong_to_the_same_owned_chunk(memgraph, shared_chunk):
    memgraph.query(
        "MATCH (s:Session {session_id: 'own'}), (c:Chunk {hash: 'shared'}) "
        "CREATE (s)-[:HAS_ACTION]->(a:Action {action_id: 'elsewhere', text: 'other chunk'}), "
        "(a)-[:HAS_CHUNK]->(:Chunk {hash: 'elsewhere'}), "
        "(:Entity {text: 'wrong chunk', embedding_model: 'test', embedding: [1.0, 0.0]})"
        "-[:MENTIONED_IN {sources: ['elsewhere']}]->(c)"
    )
    expected = memgraph.query("MATCH (n:Entity) WHERE n.text IN ['own', 'shared'] RETURN id(n) AS id ORDER BY id")
    assert memgraph.query(recall._ENTITIES, shared_chunk) == expected


def test_tied_memory_passages_use_memory_id_then_passage(memgraph):
    memgraph.query(
        "UNWIND ['z-note', 'a-note'] AS name "
        "CREATE (:Memory {user_id: 'own', memory_id: name, embedding_model: 'test', "
        "passage_embeddings: [[1.0, 0.0], [1.0, 0.0]]}) "
        "CREATE (:Memory {user_id: 'other', memory_id: 'foreign-' + name, embedding_model: 'test', "
        "passage_embeddings: [[1.0, 0.0]]})"
    )
    params = {"user": "own", "query": [1.0, 0.0], "model": "test", "k": 3}
    assert memgraph.query(recall._MEMORY_VECTORS, params) == [
        {"id": "a-note", "passage": 0},
        {"id": "a-note", "passage": 1},
        {"id": "z-note", "passage": 0},
    ]


def test_entity_fact_expansion_preserves_selected_priority_and_confidence_ties(memgraph):
    selected = memgraph.query(
        "CREATE (u:User {user_id: 'own'})-[:HAD_SESSION]->(:Session)-[:HAS_ACTION]->"
        "(:Action {action_id: 'own', text: 'evidence'}), "
        "(low:Entity {text: 'lower similarity'}), (high:Entity {text: 'higher similarity'}), "
        "(tail:Entity {text: 'target'}), "
        "(low)-[:uses {source_id: 'own', text: 'low first', confidence: 0.9}]->(tail), "
        "(low)-[:uses {source_id: 'own', text: 'low second', confidence: 0.9}]->(tail), "
        "(high)-[:uses {source_id: 'own', text: 'high first', confidence: 0.9}]->(tail), "
        "(high)-[:uses {source_id: 'own', text: 'high second', confidence: 0.9}]->(tail), "
        "(high)-[:uses {source_id: 'own', text: 'low confidence', confidence: 0.1}]->(tail) "
        "RETURN id(high) AS high, id(low) AS low"
    )[0]
    rows = memgraph.query(recall._ENTITY_FACTS, {"user": "own", "ids": [selected["high"], selected["low"]], "per": 2})
    assert [row["sentence"] for row in rows] == ["high first", "high second", "low first", "low second"]


@pytest.mark.parametrize("question", ["tooling NOT", "tooling not", 'tooling: (NOT) "*~^!', "!!!"])
def test_literal_full_text_queries_parse_for_turns_and_memory(graph, memgraph, question):
    memgraph.query(
        "CREATE (:User {user_id: 'own'})-[:HAD_SESSION]->(:Session {session_id: 's'})"
        "-[:HAS_ACTION]->(:Action {action_id: 'own', text: 'tooling NOT'}), "
        "(:User {user_id: 'other'})-[:HAD_SESSION]->(:Session {session_id: 'foreign'})"
        "-[:HAS_ACTION]->(:Action {action_id: 'foreign', text: 'tooling NOT'})"
    )
    graph.memory_store("own").create("/memories/user/tools.md", "tooling NOT")
    graph.memory_store("other").create("/memories/user/private.md", "tooling NOT")

    turns = recall._text_lane(memgraph, "own", question, 5)
    memories = recall._memories(memgraph, "own", question, RecallConfig(lanes=("memories",)), query=None, model="test")

    assert turns == (["own"] if question != "!!!" else [])
    assert [note.path for note in memories] == (["/memories/user/tools.md"] if question != "!!!" else [])
