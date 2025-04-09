import pytest 
from ..api.memgraph import MemgraphClient
from ..tools.schema import ShowSchemaInfo


def test_show_schema_info():
    uri = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

    memgraph_client = MemgraphClient(uri, user, password)

    schema_tool = ShowSchemaInfo(db=memgraph_client)
    assert "show_schema_info" in schema_tool.name

    result = schema_tool.call({})

    assert isinstance(result, str) or isinstance(result, list)  # adjust based on expected output
    schema_tool.close()