import pytest
from typing import Dict, Any, List  
from ..api.memgraph import MemgraphClient
from ..tools.schema import ShowSchemaInfo
from ..api.toolkit import Toolkit
from ..api.tool import BaseTool



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

def test_toolkit():

    toolkit = Toolkit()
    class DummyTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="dummy_tool",
                description="A dummy tool for testing",
                input_schema={}
            )

        def call(self, arguments: Dict[str, Any]) -> List[Any]:
            return ["dummy_result"]

    dummy_tool = DummyTool()
    toolkit.add(dummy_tool)

    assert toolkit.get_tool("dummy_tool") == dummy_tool
    assert len(toolkit.list_tools()) == 1

    with pytest.raises(ValueError):
        toolkit.add(dummy_tool)