from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests

from core.api.memgraph import MemgraphClient   
from langchain_memgraph.tools import QueryMemgraphTool, ShowSchemaInfoTool


class TestMemgraphIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[QueryMemgraphTool]:
        return QueryMemgraphTool

    @property
    def tool_constructor_params(self) -> dict:
        # if your tool constructor instead required initialization arguments like
        # `def __init__(self, some_arg: int):`, you would return those here
        # as a dictionary, e.g.: `return {'some_arg': 42}`
        return {"db": MemgraphClient("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns a dictionary representing the "args" of an example tool call.

        This should NOT be a ToolCall dict - i.e. it should not
        have {"name", "id", "args"} keys.
        """
        return {"query": "MATCH (n) RETURN n LIMIT 1"}


class TestSchemaInfoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[ShowSchemaInfoTool]:
        return ShowSchemaInfoTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": MemgraphClient("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns empty dict since ShowSchemaInfoTool doesn't require any parameters
        """
        return {}
