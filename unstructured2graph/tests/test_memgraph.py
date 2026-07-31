"""Unit tests for unstructured2graph.memgraph Cypher-building helpers."""

from unittest.mock import MagicMock

from lightrag_memgraph import DEFAULT_EMBEDDING_DIM
from unstructured2graph.memgraph import (
    create_nodes_from_list,
    create_unique_constraint,
    create_vector_search_index,
    promote_entity_types_to_labels,
)
from unstructured2graph.ontology import EntityType, Ontology


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


def test_promote_entity_types_to_labels_issues_one_query_per_ontology_type():
    memgraph = MagicMock()
    ontology = Ontology(entity_types=(EntityType("Person", "..."), EntityType("Organization", "...")))

    promote_entity_types_to_labels(memgraph, "base", ontology)

    # 2 per-type SET queries + 2 ontology_conformant marker queries (flag / clear).
    assert memgraph.query.call_count == 4
    first_query, first_kwargs = memgraph.query.call_args_list[0]
    assert "MATCH (n:base)" in first_query[0]
    assert "toLower(n.entity_type) = toLower($label)" in first_query[0]
    assert "SET n:Person" in first_query[0]
    assert first_kwargs["params"] == {"label": "Person"}

    second_query, second_kwargs = memgraph.query.call_args_list[1]
    assert "SET n:Organization" in second_query[0]
    assert second_kwargs["params"] == {"label": "Organization"}


def test_promote_entity_types_to_labels_never_removes_workspace_label():
    """Additive only -- must never strip the workspace label LightRAG's own
    upsert_node() relies on to re-MERGE this node on future updates. The
    ontology_conformant *property* removal (for entities that now conform)
    is fine and expected -- this only guards the workspace label itself."""
    memgraph = MagicMock()
    ontology = Ontology(entity_types=(EntityType("Person", "..."),))

    promote_entity_types_to_labels(memgraph, "base", ontology)

    all_queries = [call.args[0] for call in memgraph.query.call_args_list]
    assert not any("REMOVE n:base" in q or "DELETE" in q for q in all_queries)


def test_promote_entity_types_to_labels_flags_nonconforming_entities():
    memgraph = MagicMock()
    ontology = Ontology(entity_types=(EntityType("Person", "..."), EntityType("Organization", "...")))

    promote_entity_types_to_labels(memgraph, "base", ontology)

    flag_query = memgraph.query.call_args_list[2][0][0]
    assert "MATCH (n:base)" in flag_query
    assert "WHERE NOT (n:Person OR n:Organization)" in flag_query
    assert "SET n.ontology_conformant = false" in flag_query


def test_promote_entity_types_to_labels_clears_flag_for_conforming_entities():
    """Re-running promotion (e.g. after the ontology grows a new type) must
    clear a stale ontology_conformant=false left over from an earlier pass."""
    memgraph = MagicMock()
    ontology = Ontology(entity_types=(EntityType("Person", "..."),))

    promote_entity_types_to_labels(memgraph, "base", ontology)

    clear_query = memgraph.query.call_args_list[-1][0][0]
    assert "MATCH (n:base)" in clear_query
    assert "WHERE n:Person" in clear_query
    assert "REMOVE n.ontology_conformant" in clear_query


def test_promote_entity_types_to_labels_with_empty_ontology_flags_everything_nonconformant():
    """An empty ontology means nothing conforms -- every entity under the
    workspace label gets flagged, none rejected or deleted."""
    memgraph = MagicMock()

    promote_entity_types_to_labels(memgraph, "base", Ontology(entity_types=()))

    memgraph.query.assert_called_once()
    query = memgraph.query.call_args[0][0]
    assert "MATCH (n:base)" in query
    assert "SET n.ontology_conformant = false" in query
