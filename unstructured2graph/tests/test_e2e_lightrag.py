"""End-to-end test that exercises real LightRAG entity extraction.

Requires:
- A live Memgraph reachable at MEMGRAPH_URL (see conftest.py's `memgraph`
  fixture) -- skips if unreachable.
- MEMGRAPH_URL (and MEMGRAPH_USER/MEMGRAPH_PASSWORD/MEMGRAPH_DATABASE if
  non-default) set as literal environment variables, not just left at
  memgraph-toolbox's Python-level defaults: LightRAG's own storage backends
  read these directly and raise (not skip) if MEMGRAPH_URL is unset, even
  when a real Memgraph is reachable at the default bolt://localhost:7687.
- OPENAI_API_KEY -- skips if unset (mirrors skills-graph's
  tests/test_connector_e2e.py `requires_openai_key` convention).

Embedding stays at MemgraphLightRAGWrapper's default (Memgraph's own local
sentence-transformer, via the mage image's `embeddings` module), so only the
LLM completion call costs anything -- no separate embeddings key needed.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from lightrag_memgraph import MemgraphLightRAGWrapper
from unstructured2graph import from_texts

requires_openai_key = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)


@pytest_asyncio.fixture
async def lightrag_wrapper(memgraph, tmp_path):
    # Skip here too, not just via the test's `requires_openai_key` marker --
    # LightRAG initialization must never run (and fail confusingly) before
    # the marker's skip takes effect.
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    wrapper = MemgraphLightRAGWrapper()
    await wrapper.initialize(working_dir=str(tmp_path / "lightrag_storage"))
    yield wrapper
    await wrapper.afinalize()


@requires_openai_key
@pytest.mark.asyncio
async def test_from_texts_extracts_real_entity_and_links_mentioned_in(memgraph, lightrag_wrapper):
    grouped = await from_texts(
        ["Alice Johnson works at Acme Corp on the graph database engine."],
        memgraph,
        lightrag_wrapper,
    )

    assert len(grouped) == 1
    assert len(grouped[0]) >= 1

    workspace = lightrag_wrapper.get_lightrag().chunk_entity_relation_graph.workspace
    rows = memgraph.query(f"MATCH (e:{workspace})-[:MENTIONED_IN]->(c:Chunk) RETURN count(*) AS count")
    assert rows[0]["count"] > 0


@requires_openai_key
@pytest.mark.asyncio
async def test_from_texts_promotes_entity_type_to_label_by_default(memgraph, lightrag_wrapper):
    """DEFAULT_ONTOLOGY_PATH is applied automatically -- no ontology_path
    needed to get real Memgraph labels alongside the LightRAG workspace
    label."""
    await from_texts(
        ["Alice Johnson works at Acme Corp on the graph database engine."],
        memgraph,
        lightrag_wrapper,
    )

    workspace = lightrag_wrapper.get_lightrag().chunk_entity_relation_graph.workspace
    rows = memgraph.query(f"MATCH (p:Person:{workspace}) RETURN count(*) AS count")
    assert rows[0]["count"] > 0


@requires_openai_key
@pytest.mark.asyncio
async def test_from_texts_flags_entities_outside_a_custom_ontology(memgraph, lightrag_wrapper, tmp_path):
    """A narrow, project-specific ontology (here: only Location) still keeps
    every extracted entity and its raw entity_type -- entities outside it
    just don't get a real label, and are flagged ontology_conformant=false
    instead of being rejected."""
    ontology_file = tmp_path / "ontology.yaml"
    ontology_file.write_text(
        """
        entity_types:
          - label: Location
            description: Geographic places (cities, countries, buildings, regions)
        """
    )

    await from_texts(
        ["Alice Johnson works at Acme Corp on the graph database engine."],
        memgraph,
        lightrag_wrapper,
        ontology_path=str(ontology_file),
    )

    workspace = lightrag_wrapper.get_lightrag().chunk_entity_relation_graph.workspace
    rows = memgraph.query(
        f"MATCH (n:{workspace}) WHERE n.ontology_conformant = false RETURN n.entity_type AS entity_type, "
        "n.entity_id AS entity_id"
    )
    assert len(rows) > 0
    assert all(row["entity_type"] is not None for row in rows)

    person_rows = memgraph.query(f"MATCH (n:Person:{workspace}) RETURN count(*) AS count")
    assert person_rows[0]["count"] == 0
