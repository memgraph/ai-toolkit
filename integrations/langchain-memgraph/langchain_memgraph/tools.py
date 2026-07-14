"""Memgraph tools."""

from typing import Any

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.tools.schema import (
    EnumSchemaTool,
    NodeSchemaTool,
    RelationshipSchemaTool,
    SearchSchemaTool,
)


class BaseMemgraphTool(BaseModel):
    """
    Base tool for interacting with Memgraph.
    """

    db: Memgraph = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QueryMemgraphToolInput(BaseModel):
    query: str = Field(..., description="The query to be executed in Memgraph.")


class RunQueryTool(BaseMemgraphTool, BaseTool):
    """Tool for querying Memgraph."""

    name: str = CypherTool(db=None).get_name()
    description: str = CypherTool(db=None).get_description()
    args_schema: type[BaseModel] = _QueryMemgraphToolInput

    def _run(
        self,
        query: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        return CypherTool(db=self.db).call({"query": query})


class _SearchSchemaToolInput(BaseModel):
    pattern: str = Field(
        ..., description='A case-insensitive regex pattern to search for (e.g. "person", "pay.*ment").'
    )


class RunSearchSchemaTool(BaseMemgraphTool, BaseTool):
    """Tool for searching the graph schema by a regex pattern."""

    name: str = SearchSchemaTool(db=None).get_name()
    description: str = SearchSchemaTool(db=None).get_description()
    args_schema: type[BaseModel] = _SearchSchemaToolInput

    def _run(
        self,
        pattern: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        return SearchSchemaTool(db=self.db).call({"pattern": pattern})


class _NodeSchemaToolInput(BaseModel):
    node_labels: list[str] = Field(..., description="The labels of the node to get the details of.")


class RunNodeSchemaTool(BaseMemgraphTool, BaseTool):
    """Tool for getting the full schema definition of a node."""

    name: str = NodeSchemaTool(db=None).get_name()
    description: str = NodeSchemaTool(db=None).get_description()
    args_schema: type[BaseModel] = _NodeSchemaToolInput

    def _run(
        self,
        node_labels: list[str],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        return NodeSchemaTool(db=self.db).call({"node_labels": node_labels})


class _RelationshipSchemaToolInput(BaseModel):
    relationship_type: str = Field(..., description="The type of the relationship to get the details of.")
    start_node_labels: list[str] = Field(..., description="The labels of the start node of the relationship.")
    end_node_labels: list[str] = Field(..., description="The labels of the end node of the relationship.")


class RunRelationshipSchemaTool(BaseMemgraphTool, BaseTool):
    """Tool for getting the full schema definition of a relationship."""

    name: str = RelationshipSchemaTool(db=None).get_name()
    description: str = RelationshipSchemaTool(db=None).get_description()
    args_schema: type[BaseModel] = _RelationshipSchemaToolInput

    def _run(
        self,
        relationship_type: str,
        start_node_labels: list[str],
        end_node_labels: list[str],
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        return RelationshipSchemaTool(db=self.db).call(
            {
                "relationship_type": relationship_type,
                "start_node_labels": start_node_labels,
                "end_node_labels": end_node_labels,
            }
        )


class _EnumSchemaToolInput(BaseModel):
    enum_name: str = Field(..., description="The name of the enum to get the details of.")


class RunEnumSchemaTool(BaseMemgraphTool, BaseTool):
    """Tool for getting the schema definition of an enum."""

    name: str = EnumSchemaTool(db=None).get_name()
    description: str = EnumSchemaTool(db=None).get_description()
    args_schema: type[BaseModel] = _EnumSchemaToolInput

    def _run(
        self,
        enum_name: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        return EnumSchemaTool(db=self.db).call({"enum_name": enum_name})
