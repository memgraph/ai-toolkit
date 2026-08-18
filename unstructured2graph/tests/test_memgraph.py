"""Unit tests for unstructured2graph.memgraph.

Kept mocked deliberately -- each test here checks something a live Memgraph
run either can't easily introspect (the exact configured vector-index
dimension) or can't naturally trigger (an exception-handling branch Memgraph
itself never actually raises into). Everything that asserts on Cypher
correctness/behavior lives in test_e2e.py instead, where it's proven against
a real database rather than a plausible-looking query string.
"""

from unittest.mock import MagicMock

from lightrag_memgraph import DEFAULT_EMBEDDING_DIM
from unstructured2graph.memgraph import (
    _entity_type_to_label,
    create_unique_constraint,
    create_vector_search_index,
)


def test_create_vector_search_index_defaults_match_embedding_dim():
    """Default dimension must track lightrag_memgraph's own default, not a
    separately-maintained literal."""
    memgraph = MagicMock()

    create_vector_search_index(memgraph, "Chunk", "embedding")

    query = memgraph.query.call_args[0][0]
    assert "CREATE VECTOR INDEX vs_name ON :Chunk(embedding)" in query
    assert f"'dimension': {DEFAULT_EMBEDDING_DIM}" in query


def test_create_vector_search_index_accepts_custom_dimension_and_name():
    """A caller using a different embedding model must be able to pass a
    matching dimension and a distinct index name (e.g. for a second index
    on another label)."""
    memgraph = MagicMock()

    create_vector_search_index(memgraph, "Entity", "embedding", dimension=768, index_name="entity_vs")

    query = memgraph.query.call_args[0][0]
    assert "CREATE VECTOR INDEX entity_vs ON :Entity(embedding)" in query
    assert "'dimension': 768" in query


def test_create_unique_constraint_is_idempotent_on_repeated_calls():
    """A second call (constraint already exists) must not raise. Verified live
    (2026-08-17) that Memgraph itself never actually raises on a repeated
    identical CREATE CONSTRAINT -- this exercises the except-branch the real
    database can't be made to trigger, not a claim about Cypher correctness."""
    memgraph = MagicMock()
    memgraph.query.side_effect = [None, Exception("constraint already exists")]

    create_unique_constraint(memgraph, "Chunk", "hash")
    create_unique_constraint(memgraph, "Chunk", "hash")  # should log a warning, not raise


def test_entity_type_to_label_converts_multi_word_snake_and_space_forms():
    assert _entity_type_to_label("person") == "Person"
    assert _entity_type_to_label("natural object") == "NaturalObject"
    assert _entity_type_to_label("natural_object") == "NaturalObject"
    assert _entity_type_to_label("  Organization  ") == "Organization"


def test_entity_type_to_label_returns_none_for_unsalvageable_values():
    assert _entity_type_to_label("") is None
    assert _entity_type_to_label("   ") is None
    assert _entity_type_to_label("---") is None


def test_entity_type_to_label_returns_none_when_result_starts_with_digit():
    assert _entity_type_to_label("3d model") is None
