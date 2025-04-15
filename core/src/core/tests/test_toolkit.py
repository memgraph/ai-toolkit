import pytest
from typing import Dict, Any, List
from ..api.memgraph import MemgraphClient
from ..tools.schema import ShowSchemaInfoTool
from ..api.toolkit import Toolkit
from ..api.tool import BaseTool
from ..utils.logging import logger_init

logger = logger_init("test-toolkit")  # Set up logger for the test


def test_toolkit():
    """Test the Toolkit class."""

    toolkit = Toolkit()

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
    toolkit.add(dummy_tool)

    assert toolkit.get_tool("dummy_tool") == dummy_tool
    assert len(toolkit.list_tools()) == 1

    with pytest.raises(ValueError):
        toolkit.add(dummy_tool)
