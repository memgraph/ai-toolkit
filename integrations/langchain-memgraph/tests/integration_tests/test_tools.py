from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_memgraph.tools import (
    RunEnumSchemaTool,
    RunNodeSchemaTool,
    RunQueryTool,
    RunRelationshipSchemaTool,
    RunSearchSchemaTool,
)
from memgraph_toolbox.api.memgraph import Memgraph


class TestCypherIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunQueryTool]:
        return RunQueryTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "MATCH (n) RETURN n LIMIT 1"}


class TestSearchSchemaIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunSearchSchemaTool]:
        return RunSearchSchemaTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"pattern": ".*"}


class TestNodeSchemaIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunNodeSchemaTool]:
        return RunNodeSchemaTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"node_labels": ["Person"]}


class TestRelationshipSchemaIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunRelationshipSchemaTool]:
        return RunRelationshipSchemaTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {
            "relationship_type": "KNOWS",
            "start_node_labels": ["Person"],
            "end_node_labels": ["Person"],
        }


class TestEnumSchemaIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunEnumSchemaTool]:
        return RunEnumSchemaTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"enum_name": "Status"}
