from typing import Type

from langchain_tests.integration_tests import ToolsIntegrationTests
from langchain_memgraph.tools import (
    RunQueryTool,
    RunShowSchemaInfoTool,
    RunPageRankMemgraphTool,
    RunShowStorageInfoTool,
    RunShowConstraintInfoTool,
    RunShowIndexInfoTool,
    RunShowConfigTool,
    RunShowTriggersTool,
    RunBetweennessCentralityTool,
)
from memgraph_toolbox.api.memgraph import Memgraph


class TestSchemaInfoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[RunShowSchemaInfoTool]:
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
    def tool_constructor(self) -> Type[RunQueryTool]:
        return RunQueryTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"query": "MATCH (n) RETURN n LIMIT 1"}


class TestPageRankIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[RunPageRankMemgraphTool]:
        return RunPageRankMemgraphTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"limit": 5}


class TestStorageInfoIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> Type[RunShowStorageInfoTool]:
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
    def tool_constructor(self) -> Type[RunShowConstraintInfoTool]:
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
    def tool_constructor(self) -> Type[RunShowIndexInfoTool]:
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
    def tool_constructor(self) -> Type[RunShowConfigTool]:
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
    def tool_constructor(self) -> Type[RunShowTriggersTool]:
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
    def tool_constructor(self) -> Type[RunBetweennessCentralityTool]:
        return RunBetweennessCentralityTool

    @property
    def tool_constructor_params(self) -> dict:
        return {"db": Memgraph("bolt://localhost:7687", "", "")}

    @property
    def tool_invoke_params_example(self) -> dict:
        return {"isDirectionIgnored": True, "limit": 5}
