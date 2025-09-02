import pytest

from ..api.memgraph import Memgraph
from ..tools.betweenness_centrality import BetweennessCentralityTool
from ..tools.config import ShowConfigTool
from ..tools.constraint import ShowConstraintInfoTool
from ..tools.cypher import CypherTool
from ..tools.index import ShowIndexInfoTool
from ..tools.node_neighborhood import NodeNeighborhoodTool
from ..tools.node_vector_search import NodeVectorSearchTool
from ..tools.page_rank import PageRankTool
from ..tools.schema import ShowSchemaInfoTool
from ..tools.storage import ShowStorageInfoTool
from ..tools.trigger import ShowTriggersTool
from ..utils.logging import logger_init

logger = logger_init("test-tools")


def test_show_schema_info_tool():
    """Test the ShowSchemaInfo tool."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""

    memgraph_client = Memgraph(url=url, username=user, password=password)

    schema_tool = ShowSchemaInfoTool(db=memgraph_client)
    assert "show_schema_info" in schema_tool.name

    result = schema_tool.call({})

    assert isinstance(result, list)
    assert len(result) >= 1


def test_show_config_tool():
    """Test the ShowConfig tool."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""

    memgraph_client = Memgraph(url=url, username=user, password=password)

    config_tool = ShowConfigTool(db=memgraph_client)
    assert "show_config" in config_tool.name
    result = config_tool.call({})
    assert isinstance(result, list)
    assert len(result) >= 1


def test_index_tool():
    """Test the ShowIndexInfo tool."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""

    memgraph_client = Memgraph(url=url, username=user, password=password)

    # Create an index for testing

    memgraph_client.query("CREATE INDEX ON :Person(name)")

    index_tool = ShowIndexInfoTool(db=memgraph_client)
    assert "show_index_info" in index_tool.name
    result = index_tool.call({})

    memgraph_client.query("DROP INDEX ON :Person(name)")

    assert isinstance(result, list)

    assert len(result) >= 1


def test_storage_tool():
    """Test the ShowStorageInfo tool."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""

    memgraph_client = Memgraph(url=url, username=user, password=password)

    storage_tool = ShowStorageInfoTool(db=memgraph_client)
    assert "show_storage_info" in storage_tool.name
    result = storage_tool.call({})

    assert isinstance(result, list)
    assert len(result) >= 1


def test_show_constraint_info_tool():
    """Test the ShowConstraintInfo tool."""
    url = "bolt://localhost:7687"
    user = ""
    password = ""
    memgraph_client = Memgraph(url=url, username=user, password=password)
    # Create a sample constraint
    memgraph_client.query("CREATE CONSTRAINT ON (n:Person) ASSERT n.id IS UNIQUE")

    constraint_tool = ShowConstraintInfoTool(db=memgraph_client)
    result = constraint_tool.call({})

    memgraph_client.query("DROP CONSTRAINT ON (n:Person) ASSERT n.id IS UNIQUE")

    assert isinstance(result, list)
    assert len(result) > 0


def test_show_triggers_tool():
    """Test the ShowTriggers tool."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""
    memgraph_client = Memgraph(url=url, username=user, password=password)

    memgraph_client.query(
        """
        CREATE TRIGGER my_trigger ON () CREATE AFTER COMMIT EXECUTE
        UNWIND createdVertices AS newNodes
        SET newNodes.created = timestamp();
        """
    )

    trigger_tool = ShowTriggersTool(db=memgraph_client)
    result = trigger_tool.call({})

    memgraph_client.query("DROP TRIGGER my_trigger;")

    assert isinstance(result, list)
    assert len(result) > 0


def test_page_rank():
    """Test the PageRank tool."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""
    memgraph_client = Memgraph(url=url, username=user, password=password)

    # Create a sample graph for testing
    memgraph_client.query(
        """UNWIND range(1, 10) AS i
           CREATE (:Test {id: i})-[:LINK]->(:Test {id: i + 1}); 
        """
    )

    # Run the PageRank tool
    page_rank_tool = PageRankTool(db=memgraph_client)
    result = page_rank_tool.call({"limit": 20})
    assert isinstance(result, list)
    assert len(result) > 0


def test_cypher():
    """Test the Cypher tool."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""
    memgraph_client = Memgraph(url=url, username=user, password=password)

    cypher_tool = CypherTool(db=memgraph_client)
    result = cypher_tool.call({"query": "RETURN 0;"})
    assert isinstance(result, list)
    assert len(result) == 1


def test_cypher_date_time_serialization():
    """Test the Cypher tool with comprehensive date/time serialization."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""
    memgraph_client = Memgraph(url=url, username=user, password=password)

    cypher_tool = CypherTool(db=memgraph_client)

    # Test all temporal types supported by Memgraph
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

    # Verify all temporal types are present and properly serialized
    assert "test_date" in record
    assert "test_local_time" in record
    assert "test_local_datetime" in record
    assert "test_datetime" in record
    assert "test_duration" in record

    # Verify date serialization (Date -> "YYYY-MM-DD")
    assert record["test_date"] == "2024-01-15"

    # Verify local time serialization (LocalTime -> "HH:MM:SS.nnnnnnnnn")
    assert record["test_local_time"] == "10:30:45.000000000"

    # Verify local datetime serialization (LocalDateTime -> ISO format)
    assert record["test_local_datetime"] == "2024-01-15T10:30:45.000000000"

    # Verify datetime serialization (ZonedDateTime -> ISO format with timezone)
    assert "2024-01-15T10:30:45" in record["test_datetime"]
    assert "+01:00" in record["test_datetime"]

    # Verify duration serialization (Duration -> string representation)
    assert isinstance(record["test_duration"], str)
    assert "PT2M2.33S" in record["test_duration"]


def test_betweenness_centrality_tool():
    """Test the RunBetweennessCentralityTool."""

    url = "bolt://localhost:7687"
    user = ""
    password = ""
    memgraph_client = Memgraph(url=url, username=user, password=password)

    memgraph_client.query(
        """
        UNWIND range(1, 5) AS i
        CREATE (a:Node {id: i})-[:RELATES]->(b:Node {id: i + 1});
        """
    )

    betweenness_tool = BetweennessCentralityTool(db=memgraph_client)
    result = betweenness_tool.call({"isDirectionIgnored": True, "limit": 5})

    assert isinstance(result, list)
    assert len(result) > 0
    assert "node" in result[0]
    assert "betweenness_centrality" in result[0]


def test_node_vector_search_tool():
    """Test the NodeVectorSearch tool."""
    url = "bolt://localhost:7687"
    user = ""
    password = ""
    memgraph_client = Memgraph(url=url, username=user, password=password)

    memgraph_client.query(
        'MATCH (n:Person) WHERE "embedding" IN keys(n) DETACH DELETE n'
    )
    memgraph_client.query("DROP VECTOR INDEX my_index")
    memgraph_client.query(
        "CREATE (:Person {name: 'Alice', embedding: [1.0, 2.0, 3.0]})"
    )
    memgraph_client.query("CREATE (:Person {name: 'Bob', embedding: [1.0, 2.0, 4.0]})")
    memgraph_client.query(
        "CREATE (:Person {name: 'Charlie', embedding: [1.0, 2.0, 5.0]})"
    )
    memgraph_client.query(
        "CREATE VECTOR INDEX my_index ON :Person(embedding) WITH CONFIG {'dimension': 3, 'capacity': 1000}"
    )

    node_vector_search_tool = NodeVectorSearchTool(db=memgraph_client)
    result = node_vector_search_tool.call(
        {
            "index_name": "my_index",
            "node_label": "Person",
            "query_vector": [1.0, 2.0, 3.0],
        }
    )
    assert isinstance(result, list)
    assert len(result) == 3
    memgraph_client.query(
        'MATCH (n:Person) WHERE "embedding" IN keys(n) DETACH DELETE n'
    )
    memgraph_client.query("DROP VECTOR INDEX my_index")


def test_node_neighborhood_tool():
    """Test the NodeNeighborhood tool."""
    url = "bolt://localhost:7687"
    user = ""
    password = ""
    memgraph_client = Memgraph(url=url, username=user, password=password)

    label = "TestNodeNeighborhoodToolLabel"
    memgraph_client.query(f"MATCH (n:{label}) DETACH DELETE n;")
    memgraph_client.query(
        f"CREATE (p1:{label} {{id: 1}})-[:KNOWS]->(p2:{label} {{id: 2}}), (p2)-[:KNOWS]->(p3:{label} {{id: 3}});"
    )
    memgraph_client.query(
        f"CREATE (p4:{label} {{id: 4}})-[:KNOWS]->(p5:{label} {{id: 5}});"
    )
    ids = memgraph_client.query(
        f"MATCH (p1:{label} {{id:1}}) RETURN id(p1) AS node_id;"
    )
    assert len(ids) == 1
    node_id = ids[0]["node_id"]

    node_neighborhood_tool = NodeNeighborhoodTool(db=memgraph_client)
    result = node_neighborhood_tool.call({"node_id": node_id, "max_distance": 2})
    assert isinstance(result, list)
    assert len(result) == 2
    memgraph_client.query(f"MATCH (n:{label}) DETACH DELETE n;")
