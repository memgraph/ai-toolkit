"""Unit tests for unstructured2graph.memgraph Cypher-building helpers."""

from unittest.mock import MagicMock

from lightrag_memgraph import DEFAULT_EMBEDDING_DIM
from unstructured2graph.memgraph import create_vector_search_index


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
