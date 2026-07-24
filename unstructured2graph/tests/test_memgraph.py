"""Unit tests for unstructured2graph.memgraph Cypher-building helpers."""

from unittest.mock import MagicMock

from unstructured2graph.memgraph import create_nodes_from_list, create_unique_constraint


def test_create_nodes_from_list_defaults_to_create():
    """Without merge_key, behavior is unchanged: a plain CREATE per node."""
    memgraph = MagicMock()

    create_nodes_from_list(memgraph, [{"hash": "h1", "text": "a"}], "Chunk", 100)

    query = memgraph.query.call_args[0][0]
    assert "CREATE (n:Chunk" in query
    assert "MERGE" not in query


def test_create_nodes_from_list_merges_on_merge_key():
    """With merge_key, re-running over the same data is a no-op (MERGE)."""
    memgraph = MagicMock()

    create_nodes_from_list(memgraph, [{"hash": "h1", "text": "a"}], "Chunk", 100, merge_key="hash")

    query = memgraph.query.call_args[0][0]
    assert "MERGE (n:Chunk {hash: data.hash})" in query
    assert "ON CREATE SET n.text = data.text" in query
    assert "CREATE (n:Chunk {" not in query


def test_create_nodes_from_list_merge_key_only_property():
    """merge_key as the only property still produces valid Cypher (no dangling ON CREATE SET)."""
    memgraph = MagicMock()

    create_nodes_from_list(memgraph, [{"hash": "h1"}], "Chunk", 100, merge_key="hash")

    query = memgraph.query.call_args[0][0]
    assert "MERGE (n:Chunk {hash: data.hash})" in query
    assert "ON CREATE SET" not in query


def test_create_unique_constraint_issues_constraint_query():
    memgraph = MagicMock()

    create_unique_constraint(memgraph, "Chunk", "hash")

    query = memgraph.query.call_args[0][0]
    assert "CONSTRAINT" in query
    assert "Chunk" in query
    assert "hash" in query
    assert "UNIQUE" in query


def test_create_unique_constraint_is_idempotent_on_repeated_calls():
    """A second call (constraint already exists) must not raise."""
    memgraph = MagicMock()
    memgraph.query.side_effect = [None, Exception("constraint already exists")]

    create_unique_constraint(memgraph, "Chunk", "hash")
    create_unique_constraint(memgraph, "Chunk", "hash")  # should log a warning, not raise
