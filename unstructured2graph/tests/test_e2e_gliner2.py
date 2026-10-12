"""End-to-end test that exercises a real GLiNER2 model.

Requires:
- A live Memgraph reachable at MEMGRAPH_URL (see conftest.py's `memgraph`
  fixture) -- skips if unreachable.
- GLiNER2 and its local inference dependencies, installed with
  unstructured2graph -- skips if GLiNER2 is absent from a partial environment.

Unlike test_e2e_lightrag.py, no API key is needed: GLiNER2 runs entirely
locally. The first run downloads the model checkpoint from Hugging Face, so
this test may be slow / require network access the first time.
"""

from __future__ import annotations

import importlib.util

import pytest

from unstructured2graph import (
    Document,
    EntityType,
    Ontology,
    RelationType,
    Segment,
    enforce_relation_domain_range,
    from_documents,
    from_texts,
    ontology_report,
    promote_entity_types_to_labels,
)
from unstructured2graph.loaders import Chunk

requires_gliner2 = pytest.mark.skipif(
    importlib.util.find_spec("gliner2") is None,
    reason="gliner2 not installed",
)


@pytest.fixture(scope="module")
def gliner2_backend():
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    return GLiNER2Backend()


@requires_gliner2
def test_real_inference_extracts_typed_relations_and_exact_spans(gliner2_backend):
    """Exercise pretrained model loading and decoding without requiring a database."""
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    ontology = Ontology(
        entity_types=(
            EntityType("Person", "Human individuals"),
            EntityType("Organization", "Companies, institutions, groups"),
        ),
        relation_types=(RelationType("works_for", "Employment relationship", ("Person",), ("Organization",)),),
    )
    backend = GLiNER2Backend(ontology=ontology, model=gliner2_backend.engine.model)
    text = "Alice Johnson works for Acme Corp."
    result = backend._extract_sync(Chunk(text, "inference-employment"))

    assert result.infeasible == 0
    assert result.relations
    assert all(text[mention.start : mention.end] == mention.text for mention, _ in result.mentions)
    for label, head, tail, confidence in result.relations:
        assert label == "works_for"
        assert result.mentions[head][0].entity_type == "Person"
        assert result.mentions[tail][0].entity_type == "Organization"
        assert confidence is not None and 0 <= confidence <= 1


@requires_gliner2
def test_real_inference_extracts_values_in_a_user_turn(gliner2_backend):
    """Exercise the separate value pass on segmented user input."""
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    ontology = Ontology(
        entity_types=(
            EntityType("User", "the person speaking in the first person -- I, me, my, myself", "global"),
            EntityType("Person", "another named individual: a family member, friend, colleague", "global"),
            EntityType("Location", "a place: a city, country, neighbourhood, venue or building", "global"),
            EntityType("Duration", "a length of time: 25:50, 45 minutes, three weeks", "span"),
        ),
        relation_types=(
            RelationType("visited", "", ("User", "Person"), ("Location",)),
            RelationType("personal_best", "", ("User", "Person"), ("Duration",)),
        ),
    )
    backend = GLiNER2Backend(ontology=ontology, model=gliner2_backend.engine.model)
    text = "user: I just got back from Paris, and yesterday I ran my best 5K ever: 25:50."
    segment = Segment(0, len(text), "user", "2023-05-30T17:27:00+00:00")
    result = backend._extract_sync(Chunk(text, "inference-values", (segment,), "u1"))

    assert backend._value_schema is not None
    assert result.infeasible == 0
    assert all(text[mention.start : mention.end] == mention.text for mention, _ in result.mentions)
    assert any(mention.entity_type == "Duration" and mention.text == "25:50" for mention, _ in result.mentions)
    assert any(
        label == "personal_best" and result.mentions[tail][0].text == "25:50" for label, _, tail, _ in result.relations
    )


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
            EntityType(label="Person", description="Human individuals"),
            EntityType(label="Organization", description="Companies, institutions, groups"),
        ),
        relation_types=(
            RelationType(
                label="works_for",
                description="Employment relationship",
                start_labels=("Person",),
                end_labels=("Organization",),
            ),
        ),
    )
    backend = GLiNER2Backend(ontology=ontology)

    await from_texts(["Alice Johnson works for Acme Corp."], memgraph, backend)

    rows = memgraph.query(
        "MATCH (a:gliner2)-[r:works_for]->(b:gliner2) RETURN a.entity_type AS head, b.entity_type AS tail"
    )
    assert rows
    assert all((row["head"], row["tail"]) == ("Person", "Organization") for row in rows)


@requires_gliner2
@pytest.mark.asyncio
async def test_a_conversation_binds_the_users_facts_to_their_node_conformantly(memgraph):
    """The typed relation model on the real model: one turn per window, the
    user's own mentions bound to (:User), valid_at from the turn, and nothing
    non-conformant after the post-hoc check (#355)."""
    from unstructured2graph.gliner2_backend import GLiNER2Backend

    ontology = Ontology(
        entity_types=(
            EntityType("User", "the person speaking in the first person -- I, me, my, myself", "global"),
            EntityType("Person", "another named individual: a family member, friend, colleague", "global"),
            EntityType("Location", "a place: a city, country, neighbourhood, venue or building", "global"),
            EntityType("Duration", "a length of time: 25:50, 45 minutes, three weeks", "span"),
        ),
        relation_types=(
            RelationType("visited", "", ("User", "Person"), ("Location",)),
            RelationType("personal_best", "", ("User", "Person"), ("Duration",)),
        ),
    )
    turns = [
        (
            "user",
            "I just got back from Paris, and yesterday I ran my best 5K ever: 25:50.",
            "2023-05-30T17:27:00+00:00",
        ),
        ("assistant", "Congratulations! Paris is a great city for running.", "2023-05-30T17:27:01+00:00"),
    ]
    text, segments = "", []
    for role, body, when in turns:
        turn = f"{role}: {body}"
        start = len(text) + (2 if text else 0)
        text = f"{text}\n\n{turn}" if text else turn
        segments.append(Segment(start, start + len(turn), role, when))
    memgraph.query("MERGE (:User {user_id: 'u1'})")
    backend = GLiNER2Backend(ontology=ontology)

    grouped = await from_documents([Document(text, tuple(segments), "u1")], memgraph, backend)
    promote_entity_types_to_labels(memgraph, "gliner2", ontology)
    assert enforce_relation_domain_range(memgraph, "gliner2", ontology) == []

    rows = memgraph.query(
        "MATCH (:User {user_id: 'u1'})-[r]->(b:gliner2) RETURN type(r) AS type, b.text AS tail, toString(r.valid_at) AS valid_at"
    )
    assert rows, "expected at least one of the user's own facts on (:User)"
    assert all(row["valid_at"] == "2023-05-30T17:27:00.000000+00:00" for row in rows)
    report = ontology_report(memgraph, "gliner2", ontology, chunk_hashes=[grouped[0][0].hash])
    assert (report.nonconformant_entities, report.nonconformant_relations) == (0, 0)
    assert memgraph.query("MATCH (n:gliner2 {entity_type: 'User'}) RETURN count(n) AS n") == [{"n": 0}]
