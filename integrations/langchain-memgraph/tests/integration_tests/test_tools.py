from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_memgraph.tools import (
    RunBetweennessCentralityTool,
    RunNodeNeighborhoodTool,
    RunNodeVectorSearchTool,
    RunPageRankMemgraphTool,
    RunQueryTool,
    RunShowConfigTool,
    RunShowConstraintInfoTool,
    RunShowIndexInfoTool,
    RunShowSchemaInfoTool,
    RunShowStorageInfoTool,
    RunShowTriggersTool,
)
from memgraph_toolbox.api.memgraph import Memgraph


class TestSchemaInfoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunShowSchemaInfoTool]:
        return RunShowSchemaInfoTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns empty dict since ShowSchemaInfoTool doesn't require any parameters
        """
        return {}


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


class TestPageRankIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunPageRankMemgraphTool]:
        return RunPageRankMemgraphTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"limit": 5}


class TestStorageInfoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunShowStorageInfoTool]:
        return RunShowStorageInfoTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns empty dict since ShowStorageInfoTool doesn't require any parameters.
        """
        return {}


class TestConstraintInfoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunShowConstraintInfoTool]:
        return RunShowConstraintInfoTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns empty dict since ShowConstraintInfoTool doesn't require any parameters.
        """
        return {}


class TestIndexInfoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunShowIndexInfoTool]:
        return RunShowIndexInfoTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns empty dict since ShowIndexInfoTool doesn't require any parameters.
        """
        return {}


class TestConfigInfoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunShowConfigTool]:
        return RunShowConfigTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns empty dict since ShowConfigTool doesn't require any parameters.
        """
        return {}


class TestTriggersIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunShowTriggersTool]:
        return RunShowTriggersTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        """
        Returns empty dict since ShowTriggersTool doesn't require any parameters.
        """
        return {}


class TestBetweennessCentralityIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunBetweennessCentralityTool]:
        return RunBetweennessCentralityTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"isDirectionIgnored": True, "limit": 5}


class TestNodeNeighborhoodIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunNodeNeighborhoodTool]:
        return RunNodeNeighborhoodTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"node_id": "1", "max_distance": 2, "limit": 10}


class TestNodeVectorSearchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[RunNodeVectorSearchTool]:
        return RunNodeVectorSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"index_name": "test_index", "query_vector": [1.0, 2.0, 3.0], "limit": 5}
