"""End-to-end tests against a real Memgraph instance.

Requires Memgraph reachable at bolt://localhost:7687 (default) -- override
via MEMGRAPH_URL/MEMGRAPH_USER/MEMGRAPH_PASSWORD/MEMGRAPH_DATABASE. Skips
cleanly if unreachable (see conftest.py's `memgraph` fixture).

No LLM calls in this file -- see test_e2e_lightrag.py for the tier that
exercises real entity extraction.
"""

from __future__ import annotations

import pytest

from unstructured2graph import (
    EntityType,
    Ontology,
    compute_embeddings,
    connect_chunks_to_entities,
    create_label_index,
    create_nodes_from_list,
    create_property_index,
    create_unique_constraint,
    create_vector_search_index,
    from_texts,
    from_unstructured,
    link_nodes_in_order,
    promote_all_entity_types_to_labels,
    promote_entity_types_to_labels,
)


def test_create_unique_constraint_rejects_duplicates(memgraph):
    create_unique_constraint(memgraph, "Chunk", "hash")

    memgraph.query("CREATE (:Chunk {hash: 'h1', text: 'hello'})")
    with pytest.raises(Exception, match=r"(?i)constraint"):
        memgraph.query("CREATE (:Chunk {hash: 'h1', text: 'duplicate'})")


def test_create_label_and_property_index_do_not_raise(memgraph):
    create_label_index(memgraph, "Chunk")
    create_property_index(memgraph, "Chunk", "text")
    # No assertion beyond "didn't raise" -- Memgraph doesn't expose index
    # metadata in a form worth asserting on here.


def test_create_nodes_from_list_merge_key_is_idempotent(memgraph):
    create_unique_constraint(memgraph, "Chunk", "hash")
    nodes = [{"hash": "h1", "text": "one"}, {"hash": "h2", "text": "two"}]

    create_nodes_from_list(memgraph, nodes, "Chunk", batch_size=100, merge_key="hash")
    create_nodes_from_list(memgraph, nodes, "Chunk", batch_size=100, merge_key="hash")

    rows = memgraph.query("MATCH (n:Chunk) RETURN n.hash AS hash ORDER BY hash")
    assert [r["hash"] for r in rows] == ["h1", "h2"]


def test_create_nodes_from_list_without_merge_key_duplicates_on_rerun(memgraph):
    """Without merge_key, re-running over the same data duplicates nodes --
    the deliberate default (vs. merge_key's upsert-safe behavior above)."""
    nodes = [{"hash": "h1", "text": "one"}]

    create_nodes_from_list(memgraph, nodes, "PlainChunk", batch_size=100)
    create_nodes_from_list(memgraph, nodes, "PlainChunk", batch_size=100)

    rows = memgraph.query("MATCH (n:PlainChunk) RETURN n.hash AS hash")
    assert len(rows) == 2


def test_create_nodes_from_list_merge_key_only_property_produces_valid_node(memgraph):
    """merge_key as the only property must not produce a dangling/invalid
    ON CREATE SET clause."""
    create_unique_constraint(memgraph, "SoloKeyChunk", "hash")

    create_nodes_from_list(memgraph, [{"hash": "h1"}], "SoloKeyChunk", batch_size=100, merge_key="hash")

    rows = memgraph.query("MATCH (n:SoloKeyChunk) RETURN n.hash AS hash")
    assert [r["hash"] for r in rows] == ["h1"]


def test_link_nodes_in_order_creates_next_chain(memgraph):
    create_unique_constraint(memgraph, "Chunk", "hash")
    create_nodes_from_list(
        memgraph,
        [{"hash": "h1", "text": "a"}, {"hash": "h2", "text": "b"}, {"hash": "h3", "text": "c"}],
        "Chunk",
        batch_size=100,
        merge_key="hash",
    )

    link_nodes_in_order(
        memgraph,
        "Chunk",
        "hash",
        [{"from": "h1", "to": "h2"}, {"from": "h2", "to": "h3"}],
        "NEXT",
    )

    rows = memgraph.query(
        "MATCH (a:Chunk)-[:NEXT]->(b:Chunk) RETURN a.hash AS from_hash, b.hash AS to_hash ORDER BY from_hash"
    )
    assert [(r["from_hash"], r["to_hash"]) for r in rows] == [("h1", "h2"), ("h2", "h3")]


def test_connect_chunks_to_entities_creates_mentioned_in(memgraph):
    create_unique_constraint(memgraph, "Chunk", "hash")
    memgraph.query("CREATE (:Chunk {hash: 'h1', text: 'Alice works at Acme.'})")
    memgraph.query("CREATE (:base {name: 'Alice', file_path: 'h1'})")

    connect_chunks_to_entities(memgraph, "Chunk", "base")

    rows = memgraph.query("MATCH (e:base)-[:MENTIONED_IN]->(c:Chunk) RETURN e.name AS name, c.hash AS hash")
    assert rows == [{"name": "Alice", "hash": "h1"}]


def test_vector_search_index_and_compute_embeddings(memgraph):
    memgraph.query("CREATE (:Chunk {hash: 'h1', text: 'hello world'})")
    create_vector_search_index(memgraph, "Chunk", "embedding", index_name="test_vs_index")

    try:
        compute_embeddings(memgraph, "Chunk")
    except Exception as e:
        pytest.skip(f"Memgraph server has no embeddings module (need memgraph-mage): {e}")

    rows = memgraph.query("MATCH (n:Chunk {hash: 'h1'}) RETURN n.embedding AS embedding")
    assert rows[0]["embedding"] is not None
    assert len(rows[0]["embedding"]) > 0


@pytest.mark.asyncio
async def test_from_texts_only_chunks_creates_real_chunk_nodes(memgraph):
    grouped = await from_texts(
        ["Alice works on the graph engine.", "Bob reviews pull requests."],
        memgraph,
        only_chunks=True,
    )

    assert len(grouped) == 2
    rows = memgraph.query("MATCH (n:Chunk) RETURN n.text AS text ORDER BY text")
    assert {r["text"] for r in rows} == {"Alice works on the graph engine.", "Bob reviews pull requests."}


@pytest.mark.asyncio
async def test_from_unstructured_only_chunks_creates_real_chunk_nodes(memgraph, tmp_path):
    test_file = tmp_path / "doc.txt"
    test_file.write_text("This is a small real document used for an end-to-end chunk test.")

    grouped = await from_unstructured([str(test_file)], memgraph, only_chunks=True)

    assert len(grouped) == 1
    assert len(grouped[0]) >= 1
    rows = memgraph.query("MATCH (n:Chunk) RETURN count(n) AS count")
    assert rows[0]["count"] == len(grouped[0])


@pytest.mark.asyncio
async def test_from_unstructured_reruns_are_idempotent_not_duplicated(memgraph, tmp_path):
    """from_unstructured must self-provision its Chunk.hash constraint and
    upsert (not duplicate-insert) Chunk nodes, so re-runs over the same
    source are safe."""
    test_file = tmp_path / "doc.txt"
    test_file.write_text("Some content for idempotent end-to-end ingestion.")

    first = await from_unstructured([str(test_file)], memgraph, only_chunks=True)
    second = await from_unstructured([str(test_file)], memgraph, only_chunks=True)

    assert first == second
    rows = memgraph.query("MATCH (n:Chunk) RETURN count(n) AS count")
    assert rows[0]["count"] == len(first[0])


# ------------------------------------------------------------------
# Ontology label promotion
# ------------------------------------------------------------------


def test_promote_entity_types_to_labels_sets_matching_labels_and_flags_nonconforming(memgraph):
    memgraph.query("CREATE (:base {entity_type: 'person', name: 'Alice'})")
    memgraph.query("CREATE (:base {entity_type: 'organization', name: 'Acme'})")
    memgraph.query("CREATE (:base {entity_type: 'unknown-thing', name: 'Mystery'})")

    ontology = Ontology(entity_types=(EntityType("Person", "..."), EntityType("Organization", "...")))
    promote_entity_types_to_labels(memgraph, "base", ontology)

    rows = memgraph.query(
        "MATCH (n:base) RETURN n.name AS name, labels(n) AS labels, n.ontology_conformant AS conformant ORDER BY name"
    )
    by_name = {r["name"]: r for r in rows}
    assert "Person" in by_name["Alice"]["labels"]
    assert "base" in by_name["Alice"]["labels"]  # workspace label never removed
    assert by_name["Alice"]["conformant"] is None  # never flagged / cleared for conforming entities
    assert "Organization" in by_name["Acme"]["labels"]
    assert by_name["Mystery"]["labels"] == ["base"]  # not rejected, not deleted, just unlabeled
    assert by_name["Mystery"]["conformant"] is False


def test_promote_entity_types_to_labels_rerun_clears_stale_flag_and_is_idempotent(memgraph):
    """Re-running after the ontology grows a new type must clear a stale
    ontology_conformant=false left over from an earlier, narrower pass, and a
    second run with the same ontology must not error or double-apply a label."""
    memgraph.query("CREATE (:base {entity_type: 'organization', name: 'Acme'})")

    promote_entity_types_to_labels(memgraph, "base", Ontology(entity_types=(EntityType("Person", "..."),)))
    row = memgraph.query("MATCH (n:base {name: 'Acme'}) RETURN n.ontology_conformant AS conformant")[0]
    assert row["conformant"] is False

    grown_ontology = Ontology(entity_types=(EntityType("Person", "..."), EntityType("Organization", "...")))
    promote_entity_types_to_labels(memgraph, "base", grown_ontology)
    row = memgraph.query("MATCH (n:base {name: 'Acme'}) RETURN labels(n) AS labels, n.ontology_conformant AS c")[0]
    assert "Organization" in row["labels"]
    assert row["c"] is None

    promote_entity_types_to_labels(memgraph, "base", grown_ontology)
    row = memgraph.query("MATCH (n:base {name: 'Acme'}) RETURN labels(n) AS labels")[0]
    assert row["labels"].count("Organization") == 1


def test_promote_entity_types_to_labels_with_empty_ontology_flags_everything(memgraph):
    memgraph.query("CREATE (:base {entity_type: 'person', name: 'Alice'})")

    promote_entity_types_to_labels(memgraph, "base", Ontology(entity_types=()))

    row = memgraph.query(
        "MATCH (n:base {name: 'Alice'}) RETURN labels(n) AS labels, n.ontology_conformant AS conformant"
    )[0]
    assert row["labels"] == ["base"]
    assert row["conformant"] is False


def test_promote_all_entity_types_to_labels_discovers_and_promotes_without_conformance_flag(memgraph):
    memgraph.query("CREATE (:base {entity_type: 'person', name: 'Alice'})")
    memgraph.query("CREATE (:base {entity_type: 'natural object', name: 'Rock'})")
    memgraph.query("CREATE (:base {entity_type: '---', name: 'Unsanitizable'})")

    promote_all_entity_types_to_labels(memgraph, "base")

    rows = memgraph.query(
        "MATCH (n:base) RETURN n.name AS name, labels(n) AS labels, n.ontology_conformant AS conformant ORDER BY name"
    )
    by_name = {r["name"]: r for r in rows}
    assert "Person" in by_name["Alice"]["labels"]
    assert "NaturalObject" in by_name["Rock"]["labels"]
    assert by_name["Unsanitizable"]["labels"] == ["base"]  # skipped, left as-is
    assert all(r["conformant"] is None for r in rows)  # no ontology here, nothing to flag against
