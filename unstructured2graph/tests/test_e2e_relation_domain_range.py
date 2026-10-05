"""The post-hoc half of the typed relation model, against a real Memgraph."""

import pytest

from hygm import ValidationCategory, ValidationSeverity
from unstructured2graph import (
    Document,
    EntityType,
    Ontology,
    RelationType,
    Segment,
    enforce_relation_domain_range,
    from_documents,
    ontology_report,
    promote_entity_types_to_labels,
)
from unstructured2graph.gliner2_backend import GLiNER2Backend

from .gliner2_fakes import FakeEngine

ONTOLOGY = Ontology(
    entity_types=(
        EntityType("User", "the user", "global"),
        EntityType("Person", "someone else", "global"),
        EntityType("Location", "a place", "global"),
        EntityType("Organization", "an institution", "global"),
    ),
    relation_types=(
        RelationType("visited", "", ("User", "Person"), ("Location",)),
        RelationType("works_for", "", ("User", "Person"), ("Organization",)),
        RelationType("mentions"),
    ),
)


def _graph(memgraph, edges):
    """Workspace `ws` entities by text, typed, and the given (head, type, tail) edges between them."""
    types = {"alice": "Person", "paris": "Location", "moma": "Organization"}
    for text, entity_type in types.items():
        memgraph.query(
            "CREATE (:ws {entity_id: $text, text: $text, entity_type: $entity_type})",
            params={"text": text, "entity_type": entity_type},
        )
    for head, relation, tail in edges:
        memgraph.query(
            f"MATCH (a:ws {{entity_id: $head}}), (b:ws {{entity_id: $tail}}) CREATE (a)-[:{relation} {{chunk: 'c1'}}]->(b)",
            params={"head": head, "tail": tail},
        )
    promote_entity_types_to_labels(memgraph, "ws", ONTOLOGY)


def _flags(memgraph):
    return memgraph.query(
        "MATCH (a)-[r]->(b) RETURN a.text AS head, type(r) AS type, b.text AS tail, r.ontology_conformant AS flag "
        "ORDER BY head, type, tail"
    )


def test_a_relationship_outside_its_domain_range_is_flagged_and_kept(memgraph):
    _graph(memgraph, [("alice", "visited", "paris"), ("alice", "visited", "moma")])
    issues = enforce_relation_domain_range(memgraph, "ws", ONTOLOGY)
    assert _flags(memgraph) == [
        {"head": "alice", "type": "visited", "tail": "moma", "flag": False},
        {"head": "alice", "type": "visited", "tail": "paris", "flag": None},
    ]
    assert len(issues) == 1
    assert issues[0].severity is ValidationSeverity.WARNING
    assert issues[0].details == {"relation_type": "visited", "count": 1}
    assert issues[0].actual == (("Person",), ("Organization",))


def test_a_rerun_clears_a_flag_that_no_longer_applies(memgraph):
    _graph(memgraph, [("alice", "visited", "moma")])
    enforce_relation_domain_range(memgraph, "ws", ONTOLOGY)
    widened = Ontology(
        entity_types=ONTOLOGY.entity_types,
        relation_types=(RelationType("visited", "", ("User", "Person"), ("Location", "Organization")),),
    )
    assert enforce_relation_domain_range(memgraph, "ws", widened) == []
    assert _flags(memgraph)[0]["flag"] is None


def test_an_unconstrained_relation_type_is_never_flagged(memgraph):
    _graph(memgraph, [("paris", "mentions", "moma")])
    assert enforce_relation_domain_range(memgraph, "ws", ONTOLOGY) == []
    assert _flags(memgraph)[0]["flag"] is None


def test_scoping_by_chunk_leaves_other_chunks_alone(memgraph):
    _graph(memgraph, [("alice", "visited", "moma")])
    assert enforce_relation_domain_range(memgraph, "ws", ONTOLOGY, chunk_hashes=["other"]) == []
    assert _flags(memgraph)[0]["flag"] is None


def test_ontology_report_counts_flags_and_zero_instance_types(memgraph):
    _graph(memgraph, [("alice", "visited", "paris"), ("alice", "visited", "moma")])
    enforce_relation_domain_range(memgraph, "ws", ONTOLOGY)
    report = ontology_report(memgraph, "ws", ONTOLOGY)
    assert (report.relationships, report.declared_relationships, report.nonconformant_relations) == (2, 2, 1)
    assert report.nonconformant_entities == 0
    assert report.zero_instance_relation_types == ("works_for", "mentions")
    assert [issue.category for issue in report.issues] == [ValidationCategory.COVERAGE]


def test_ontology_report_tells_untyped_relations_from_none_at_all(memgraph):
    _graph(memgraph, [("alice", "DIRECTED", "paris")])
    untyped = ontology_report(memgraph, "ws", ONTOLOGY)
    assert (untyped.relationships, untyped.declared_relationships) == (1, 0)
    assert "none of a declared relation type" in untyped.issues[0].message

    memgraph.query("MATCH ()-[r:DIRECTED]->() DELETE r")
    empty = ontology_report(memgraph, "ws", ONTOLOGY)
    assert empty.relationships == 0
    assert empty.issues[0].message == "No relationships extracted"


@pytest.mark.asyncio
async def test_gliner2_output_is_conformant_by_construction(memgraph):
    """One specification, two compilations (#348): a post-hoc violation on GLiNER2's
    output would be a bug, so this asserts zero -- through the resolver's re-typing too."""
    memgraph.query("MERGE (:User {user_id: 'u1'})")
    text = "user: I visited Paris and Kahlo works for MoMA"
    engine = FakeEngine(
        surfaces={"I": "User", "Paris": "Location", "Kahlo": "User", "MoMA": "Organization"},
        relations=[("visited", "I", "Paris", 0.9), ("works_for", "Kahlo", "MoMA", 0.9)],
    )
    backend = GLiNER2Backend(ontology=ONTOLOGY, model=engine)
    document = Document(text, (Segment(0, len(text), "user", "2023-05-30T17:27:00+00:00"),), user_id="u1")
    chunks = await from_documents([document], memgraph, backend)
    promote_entity_types_to_labels(memgraph, "gliner2", ONTOLOGY)
    assert enforce_relation_domain_range(memgraph, "gliner2", ONTOLOGY) == []
    report = ontology_report(memgraph, "gliner2", ONTOLOGY, chunk_hashes=[chunks[0][0].hash])
    assert (report.nonconformant_entities, report.nonconformant_relations) == (0, 0)
    assert report.declared_relationships == 2


def test_a_violation_from_outside_the_workspace_is_still_flagged(memgraph):
    """Guards the (a:ws OR b:ws) planner bug: an edge from the sessions-graph-owned
    (:User) node has only one workspace endpoint, and must still be checked."""
    _graph(memgraph, [])
    memgraph.query("CREATE (:User {user_id: 'u1', text: 'u1'})")
    memgraph.query("MATCH (u:User), (b:ws {entity_id: 'paris'}) CREATE (u)-[:works_for {chunk: 'c1'}]->(b)")
    issues = enforce_relation_domain_range(memgraph, "ws", ONTOLOGY)
    assert [issue.details for issue in issues] == [{"relation_type": "works_for", "count": 1}]
    assert ontology_report(memgraph, "ws", ONTOLOGY).nonconformant_relations == 1
