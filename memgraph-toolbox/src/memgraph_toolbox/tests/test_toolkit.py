from typing import Any, Dict, List

import pytest

from ..api.toolkit import BaseToolkit
from ..api.tool import BaseTool
from ..utils.logging import logger_init

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
