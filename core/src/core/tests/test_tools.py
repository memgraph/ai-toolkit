import pytest
from ..api.memgraph import MemgraphClient
from ..tools.schema import ShowSchemaInfo
from ..tools.config import ShowConfig
from ..tools.index import ShowIndexInfo
from ..tools.storage import ShowStorageInfo
from ..tools.constraint import ShowConstraintInfo
from ..tools.trigger import ShowTriggers
from ..utils.logging import logger_init  # Import centralized logger

logger = logger_init("test-tools")  # Set up logger for the test


def test_show_schema_info_tool():
    """Test the ShowSchemaInfo tool."""

    uri = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

    memgraph_client = MemgraphClient(uri=uri, username=user, password=password)

    schema_tool = ShowSchemaInfo(db=memgraph_client)
    assert "show_schema_info" in schema_tool.name

    result = schema_tool.call({})

    assert isinstance(result, list)
    assert len(result) >= 1
    schema_tool.close()


def test_config_tool():
    """Test the ShowConfig tool."""

    uri = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

    memgraph_client = MemgraphClient(uri=uri, username=user, password=password)

    config_tool = ShowConfig(db=memgraph_client)
    assert "show_config" in config_tool.name
    result = config_tool.call({})
    assert isinstance(result, list)
    assert len(result) >= 1
    config_tool.close()


def test_index_tool():
    """Test the ShowIndexInfo tool."""

    uri = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

    memgraph_client = MemgraphClient(uri=uri, username=user, password=password)

    # Create an index for testing

    memgraph_client.query("CREATE INDEX ON :Person(name)")

    index_tool = ShowIndexInfo(db=memgraph_client)
    assert "show_index_info" in index_tool.name
    result = index_tool.call({})

    memgraph_client.query("DROP INDEX ON :Person(name)")

    assert isinstance(result, list)

    assert len(result) >= 1


    index_tool.close()


def test_storage_tool():
    """Test the ShowStorageInfo tool."""

    uri = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

    memgraph_client = MemgraphClient(uri=uri, username=user, password=password)

    storage_tool = ShowStorageInfo(db=memgraph_client)
    assert "show_storage_info" in storage_tool.name
    result = storage_tool.call({})

    assert isinstance(result, list)
    assert len(result) >= 1
    storage_tool.close()


def test_show_constraint_info_tool():
    """Test the ShowConstraintInfo tool."""
    uri = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"
    memgraph_client = MemgraphClient(uri=uri, username=user, password=password)
    # Create a sample constraint
    memgraph_client.query("CREATE CONSTRAINT ON (n:Person) ASSERT n.id IS UNIQUE")

    constraint_tool = ShowConstraintInfo(db=memgraph_client)
    result = constraint_tool.call({})

    memgraph_client.query("DROP CONSTRAINT ON (n:Person) ASSERT n.id IS UNIQUE")

    assert isinstance(result, list)
    assert len(result) > 0

    constraint_tool.close()


def test_show_triggers_tool():
    """Test the ShowTriggers tool."""

    uri = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"
    memgraph_client = MemgraphClient(uri=uri, username=user, password=password)

    memgraph_client.query(
        """
        CREATE TRIGGER my_trigger ON () CREATE AFTER COMMIT EXECUTE
        UNWIND createdVertices AS newNodes
        SET newNodes.created = timestamp();
        """
    )

    trigger_tool = ShowTriggers(db=memgraph_client)
    result = trigger_tool.call({})

    memgraph_client.query("DROP TRIGGER my_trigger;")

    assert isinstance(result, list)
    assert len(result) > 0

    trigger_tool.close()
