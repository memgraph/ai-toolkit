"""End-to-end test that exercises a real GLiNER2 model.

Requires:
- A live Memgraph reachable at MEMGRAPH_URL (see conftest.py's `memgraph`
  fixture) -- skips if unreachable.
- The optional `gliner2` dependency (`pip install unstructured2graph[gliner2]`)
  -- skips if not installed, so the default test suite never needs it.

Unlike test_e2e_lightrag.py, no API key is needed: GLiNER2 runs entirely
locally. The first run downloads the model checkpoint from Hugging Face, so
this test may be slow / require network access the first time.
"""

from __future__ import annotations

import importlib.util

import pytest

from unstructured2graph import EntityType, Ontology, RelationType, from_texts

requires_gliner2 = pytest.mark.skipif(
    importlib.util.find_spec("gliner2") is None,
    reason="gliner2 not installed",
)


@pytest.fixture
def gliner2_backend():
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    return GLiNER2Backend()


@requires_gliner2
@pytest.mark.asyncio
async def test_from_texts_extracts_real_entity_and_links_mentioned_in(memgraph, gliner2_backend):
    grouped = await from_texts(
        ["Alice Johnson works at Acme Corp on the graph database engine."],
        memgraph,
        gliner2_backend,
    )

    assert len(grouped) == 1
    assert len(grouped[0]) >= 1

    rows = memgraph.query("MATCH (e:gliner2)-[:MENTIONED_IN]->(c:Chunk) RETURN count(*) AS count")
    assert rows[0]["count"] > 0

    entity_rows = memgraph.query("MATCH (e:gliner2) RETURN e.entity_type AS entity_type, e.file_path AS file_path")
    assert len(entity_rows) > 0
    assert all(row["file_path"] is not None for row in entity_rows)


@requires_gliner2
@pytest.mark.asyncio
async def test_from_texts_extracts_typed_relation_with_custom_ontology(memgraph):
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    ontology = Ontology(
        entity_types=(
            EntityType(label="person", description="Human individuals"),
            EntityType(label="organization", description="Companies, institutions, groups"),
        ),
        relation_types=(RelationType(label="works_for", description="Employment relationship"),),
    )
    backend = GLiNER2Backend(ontology=ontology)

    await from_texts(
        ["Alice Johnson works for Acme Corp."],
        memgraph,
        backend,
    )

    rows = memgraph.query("MATCH (:gliner2)-[r:works_for]->(:gliner2) RETURN count(r) AS count")
    assert rows[0]["count"] > 0
