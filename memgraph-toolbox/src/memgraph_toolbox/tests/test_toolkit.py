from typing import Any, Dict, List

import pytest

from ..api.toolkit import BaseToolkit
from ..memgraph_toolkit import MemgraphToolkit
from ..api.tool import BaseTool
from ..utils.logging import logger_init
from ..api.memgraph import Memgraph

logger = logger_init("test-toolkit")  # Set up logger for the test


def test_toolkit():
    """Test the Toolkit class."""

    toolkit = BaseToolkit()

    class DummyTool(BaseTool):
        def __init__(self):
            super().__init__(
                name="dummy_tool",
                description="A dummy tool for testing",
                input_schema={},
            )

        def call(self, arguments: Dict[str, Any]) -> List[Any]:
            return ["dummy_result"]

    dummy_tool = DummyTool()
    toolkit.add_tool(dummy_tool)

    assert toolkit.get_tool("dummy_tool") == dummy_tool
    assert len(toolkit.get_all_tools()) == 1

    with pytest.raises(ValueError):
        toolkit.add_tool(dummy_tool)


def test_memgraph_toolkit():
    """Test the Memgraph Toolkit."""

    url = "bolt://localhost:7687"
    user = "memgraph"
    password = "memgraph"

    memgraph_client = Memgraph(url=url, username=user, password=password)

    toolkit = MemgraphToolkit(db=memgraph_client)

    tools = toolkit.get_all_tools()

    # Check if we have all 9 tools
    assert len(tools) == 9

    # Check for specific tool names
    tool_names = [tool.name for tool in tools]
    expected_tools = [
        "run_betweenness_centrality",
        "show_config",
        "show_constraint_info",
        "run_cypher",
        "show_index_info",
        "page_rank",
        "show_schema_info",
        "show_storage_info",
        "show_triggers",
    ]

    for expected_tool in expected_tools:
        assert expected_tool in tool_names
