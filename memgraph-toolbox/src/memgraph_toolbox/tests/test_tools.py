import contextlib

import pytest

from ..api.memgraph import Memgraph
from ..tools.cypher import CypherTool
from ..tools.graph_cypher import GraphCypherTool
from ..tools.schema import (
    EnumSchemaTool,
    NodeSchemaTool,
    RelationshipSchemaTool,
    SearchSchemaTool,
)


@pytest.fixture()
def db():
    client = Memgraph(url="bolt://localhost:7687", username="", password="")
    yield client
    client.close()


@pytest.fixture()
def schema_graph(db):
    """Create a small graph so SHOW SCHEMA INFO has node/edge types to report."""
    db.query("MATCH (n) DETACH DELETE n")
    db.query("CREATE (a:Person {name: 'Alice', age: 30})-[:KNOWS {since: 2020}]->(b:Person {name: 'Bob', age: 25})")
    db.query("MATCH (b:Person {name: 'Bob'}) CREATE (b)-[:WORKS_AT {role: 'engineer'}]->(:Company {name: 'Memgraph'})")
    yield
    db.query("MATCH (n) DETACH DELETE n")


def test_cypher(db):
    """Test the Cypher tool."""
    cypher_tool = CypherTool(db=db)
    result = cypher_tool.call({"query": "RETURN 0;"})
    assert isinstance(result, list)
    assert len(result) == 1


def test_cypher_date_time_serialization(db):
    """Test the Cypher tool with comprehensive date/time serialization."""
    cypher_tool = CypherTool(db=db)

    query = """
    RETURN
        date('2024-01-15') AS test_date,
        localTime('10:30:45') AS test_local_time,
        localDateTime('2024-01-15T10:30:45') AS test_local_datetime,
        datetime('2024-01-15T10:30:45+01:00') AS test_datetime,
        duration('PT2M2.33S') AS test_duration
    """

    result = cypher_tool.call({"query": query})

    assert isinstance(result, list)
    assert len(result) == 1

    record = result[0]

    assert "test_date" in record
    assert "test_local_time" in record
    assert "test_local_datetime" in record
    assert "test_datetime" in record
    assert "test_duration" in record

    assert record["test_date"] == "2024-01-15"
    assert record["test_local_time"] == "10:30:45.000000000"
    assert record["test_local_datetime"] == "2024-01-15T10:30:45.000000000"
    assert "2024-01-15T10:30:45" in record["test_datetime"]
    assert "+01:00" in record["test_datetime"]
    assert isinstance(record["test_duration"], str)
    assert "PT2M2.33S" in record["test_duration"]


def test_graph_cypher_projects_nodes_and_edges(db, schema_graph):
    """A `RETURN a, r, b` projects connected nodes and a typed edge."""
    tool = GraphCypherTool(db=db)
    result = tool.call({"query": "MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person) RETURN a, r, b"})

    assert isinstance(result, dict)
    assert "nodes" in result and "relationships" in result

    nodes = result["nodes"]
    rels = result["relationships"]
    assert len(nodes) == 2
    assert len(rels) == 1

    names = {n["properties"].get("name") for n in nodes}
    assert names == {"Alice", "Bob"}
    assert all("Person" in n["labels"] for n in nodes)

    edge = rels[0]
    assert edge["type"] == "KNOWS"
    assert edge["properties"].get("since") == 2020
    node_ids = {n["id"] for n in nodes}
    assert edge["start"] in node_ids
    assert edge["end"] in node_ids


def test_graph_cypher_deduplicates_shared_nodes(db, schema_graph):
    """A node appearing in several rows is emitted once, keyed on element id."""
    tool = GraphCypherTool(db=db)
    result = tool.call({"query": "MATCH (a)-[r]->(b) RETURN a, r, b"})

    assert isinstance(result, dict)
    assert len(result["nodes"]) == 3
    assert len(result["relationships"]) == 2
    ids = [n["id"] for n in result["nodes"]]
    assert len(ids) == len(set(ids)), "nodes must be deduplicated"


def test_graph_cypher_projects_path(db, schema_graph):
    """A returned path is unrolled into its nodes and relationships."""
    tool = GraphCypherTool(db=db)
    result = tool.call({"query": "MATCH p = (:Person)-[:KNOWS]->(:Person) RETURN p"})

    assert isinstance(result, dict)
    assert len(result["nodes"]) == 2
    assert len(result["relationships"]) == 1
    assert result["relationships"][0]["type"] == "KNOWS"


def test_graph_cypher_scalar_has_no_entities(db, schema_graph):
    """A query returning only scalars yields an empty projection, not an error."""
    tool = GraphCypherTool(db=db)
    result = tool.call({"query": "RETURN 1 AS n"})

    assert result == {"nodes": [], "relationships": []}


def test_graph_cypher_invalid_query_returns_error(db):
    """A syntactically invalid query returns the list-shaped error convention."""
    tool = GraphCypherTool(db=db)
    result = tool.call({"query": "NOT VALID CYPHER"})

    assert isinstance(result, list)
    assert len(result) == 1
    assert "error" in result[0]


def test_get_node_schema_found(db, schema_graph):
    tool = NodeSchemaTool(db=db)
    result = tool.call({"node_labels": ["Person"]})

    assert isinstance(result, dict)
    assert "node" in result
    node = result["node"]
    assert "Person" in node["labels"]

    prop_keys = [p["key"] for p in node.get("properties", [])]
    assert "name" in prop_keys
    assert "age" in prop_keys

    assert "relationships" in result
    rels = result["relationships"]
    assert isinstance(rels["outbound"], list)
    assert isinstance(rels["inbound"], list)


def test_get_node_schema_not_found(db, schema_graph):
    tool = NodeSchemaTool(db=db)
    result = tool.call({"node_labels": ["NonExistent"]})

    assert isinstance(result, list)
    assert len(result) == 1
    assert "No node found" in result[0]["text"]


def test_get_relationship_schema_found(db, schema_graph):
    tool = RelationshipSchemaTool(db=db)
    result = tool.call(
        {
            "relationship_type": "KNOWS",
            "start_node_labels": ["Person"],
            "end_node_labels": ["Person"],
        }
    )

    assert isinstance(result, dict)
    assert "relationship" in result
    edge = result["relationship"]
    assert edge["type"] == "KNOWS"

    prop_keys = [p["key"] for p in edge.get("properties", [])]
    assert "since" in prop_keys


def test_get_relationship_schema_not_found(db, schema_graph):
    tool = RelationshipSchemaTool(db=db)
    result = tool.call(
        {
            "relationship_type": "FAKE_REL",
            "start_node_labels": ["Person"],
            "end_node_labels": ["Person"],
        }
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert "No relationship found" in result[0]["text"]


def test_get_enum_schema_found(db, schema_graph):
    with contextlib.suppress(Exception):
        # There is no way to remove enums from Memgraph
        db.query("CREATE ENUM Status VALUES { Active, Inactive, Pending }")

    tool = EnumSchemaTool(db=db)
    result = tool.call({"enum_name": "Status"})

    assert isinstance(result, dict)
    assert "enum" in result
    assert result["enum"]["name"] == "Status"
    assert set(result["enum"]["values"]) == {"Active", "Inactive", "Pending"}


def test_get_enum_schema_not_found(db, schema_graph):
    tool = EnumSchemaTool(db=db)
    result = tool.call({"enum_name": "NonExistentEnum"})

    assert isinstance(result, list)
    assert len(result) == 1
    assert "No enum found" in result[0]["text"]


def test_search_schema_matches_node_label(db, schema_graph):
    tool = SearchSchemaTool(db=db)
    result = tool.call({"pattern": "Person"})

    assert isinstance(result, list)
    assert len(result) > 0
    labels_in_matches = [m.get("labels", []) for m in result if m.get("type") == "node-match"]
    assert any("Person" in labels for labels in labels_in_matches)


def test_search_schema_matches_relationship_type(db, schema_graph):
    tool = SearchSchemaTool(db=db)
    result = tool.call({"pattern": "KNOWS"})

    assert isinstance(result, list)
    assert len(result) > 0
    edge_types = [m.get("edge_type") for m in result if m.get("type") == "edge-match"]
    assert "KNOWS" in edge_types


def test_search_schema_matches_property_key(db, schema_graph):
    tool = SearchSchemaTool(db=db)
    result = tool.call({"pattern": "name"})

    assert isinstance(result, list)
    assert len(result) > 0


def test_search_schema_regex_pattern(db, schema_graph):
    tool = SearchSchemaTool(db=db)
    result = tool.call({"pattern": "KNOW.*"})

    assert isinstance(result, list)
    assert len(result) > 0


def test_search_schema_no_matches(db, schema_graph):
    tool = SearchSchemaTool(db=db)
    result = tool.call({"pattern": "zzz_no_match_zzz"})

    assert isinstance(result, list)
    assert len(result) == 1
    assert "No schema elements matching" in result[0]["text"]
