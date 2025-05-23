import pytest

from ..api.memgraph import Memgraph
from ..tools.betweenness_centrality import BetweennessCentralityTool
from ..tools.config import ShowConfigTool
from ..tools.constraint import ShowConstraintInfoTool
from ..tools.cypher import CypherTool
from ..tools.index import ShowIndexInfoTool
from ..tools.page_rank import PageRankTool
from ..tools.schema import ShowSchemaInfoTool
from ..tools.storage import ShowStorageInfoTool
from ..tools.trigger import ShowTriggersTool
from ..utils.logging import logger_init

logger = logger_init("test-tools")


def test_show_schema_info_tool():
    """Test the ShowSchemaInfo tool."""

    url = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

    memgraph_client = Memgraph(url=url, username=user, password=password)

    schema_tool = ShowSchemaInfoTool(db=memgraph_client)
    assert "show_schema_info" in schema_tool.name

    result = schema_tool.call({})

    assert isinstance(result, list)
    assert len(result) >= 1


def test_show_config_tool():
    """Test the ShowConfig tool."""

    url = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

    memgraph_client = Memgraph(url=url, username=user, password=password)

    config_tool = ShowConfigTool(db=memgraph_client)
    assert "show_config" in config_tool.name
    result = config_tool.call({})
    assert isinstance(result, list)
    assert len(result) >= 1


def test_index_tool():
    """Test the ShowIndexInfo tool."""

    url = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

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
    user = "memgraph"
    password = "memgraph"

    memgraph_client = Memgraph(url=url, username=user, password=password)

    storage_tool = ShowStorageInfoTool(db=memgraph_client)
    assert "show_storage_info" in storage_tool.name
    result = storage_tool.call({})

    assert isinstance(result, list)
    assert len(result) >= 1


def test_show_constraint_info_tool():
    """Test the ShowConstraintInfo tool."""
    url = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"
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
    user = "memgraph"
    password = "memgraph"
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
    user = "memgraph"
    password = "memgraph"
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
    user = "memgraph"
    password = "memgraph"
    memgraph_client = Memgraph(url=url, username=user, password=password)

    cypher_tool = CypherTool(db=memgraph_client)
    result = cypher_tool.call({"query": "RETURN 0;"})
    assert isinstance(result, list)
    assert len(result) == 1


def test_betweenness_centrality_tool():
    """Test the RunBetweennessCentralityTool."""

    url = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"
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
