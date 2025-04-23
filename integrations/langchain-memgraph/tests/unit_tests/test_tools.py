from typing import Type

from langchain_tests.unit_tests import ToolsUnitTests

from memgraph_toolbox.api.memgraph import Memgraph
from langchain_memgraph.tools import RunQueryTool
from langchain_memgraph.tools import RunShowStorageInfoTool


class TestMemgraphIntegration(ToolsUnitTests):
    @property
    def tool_constructor(self) -> Type[RunQueryTool]:
        return RunQueryTool

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"query": "MATCH (n) RETURN n LIMIT 1"}
