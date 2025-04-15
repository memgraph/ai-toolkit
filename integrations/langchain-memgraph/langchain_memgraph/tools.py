"""Memgraph tools."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from core.api.memgraph import MemgraphClient
from core.tools.schema import ShowSchemaInfoTool
from core.tools.cypher import CypherTool


class BaseMemgraphTool(BaseModel):
    """
    Base tool for interacting with Memgraph.
    """

    db: MemgraphClient = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QueryMemgraphToolInput(BaseModel):
    """
    Input query for Memgraph Query tool.
    """

    query: str = Field(..., description="The query to be executed in Memgraph.")


class RunQueryMemgraphTool(BaseMemgraphTool, BaseTool): 
    """Tool for querying Memgraph."""

    name: str = CypherTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = CypherTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Type[BaseModel] = _QueryMemgraphToolInput
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        return CypherTool(
            db=self.db,
        ).call({"query": query})


class RunShowSchemaInfoTool(BaseMemgraphTool, BaseTool): 
    """Tool for retrieving schema information from Memgraph."""

    name: str = ShowSchemaInfoTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = ShowSchemaInfoTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Type[BaseModel] = BaseModel
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Run the tool to get schema information."""

        schema_info = ShowSchemaInfoTool(
            db=self.db,
        )
        result = schema_info.call({})

        return result



