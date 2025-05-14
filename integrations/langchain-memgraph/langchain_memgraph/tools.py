"""Memgraph tools."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.page_rank import PageRankTool
from memgraph_toolbox.tools.config import ShowConfigTool
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.tools.schema import ShowSchemaInfoTool
from memgraph_toolbox.tools.storage import ShowStorageInfoTool
from memgraph_toolbox.tools.trigger import ShowTriggersTool
from memgraph_toolbox.tools.index import ShowIndexInfoTool
from memgraph_toolbox.tools.betweenness_centrality import BetweennessCentralityTool
from memgraph_toolbox.tools.constraint import ShowConstraintInfoTool
from memgraph_toolbox.utils.logging import logger_init


class BaseMemgraphTool(BaseModel):
    """
    Base tool for interacting with Memgraph.
    """

    db: Memgraph = Field(exclude=True)

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class _QueryMemgraphToolInput(BaseModel):
    """
    Input query for Memgraph Query tool.
    """

    query: str = Field(..., description="The query to be executed in Memgraph.")


class RunQueryTool(BaseMemgraphTool, BaseTool):
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

    args_schema: Optional[Type[BaseModel]] = None
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


class RunShowStorageInfoTool(BaseMemgraphTool, BaseTool):
    """Tool for retrieving storage information from Memgraph."""

    name: str = ShowStorageInfoTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = ShowStorageInfoTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Optional[Type[BaseModel]] = None
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Run the tool to get storage information."""

        storage_info = ShowStorageInfoTool(
            db=self.db,
        )
        result = storage_info.call({})

        return result


class _PageRankMemgraphToolInput(BaseModel):
    """
    Input schema for the PageRank Memgraph tool.
    """

    limit: int = Field(10, description="Number of nodes to return.")


class RunPageRankMemgraphTool(BaseMemgraphTool, BaseTool):
    """Tool for running PageRank on Memgraph."""

    name: str = PageRankTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = PageRankTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Type[BaseModel] = _PageRankMemgraphToolInput
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        return PageRankTool(
            db=self.db,
        ).call({"limit": limit})


class RunShowConstraintInfoTool(BaseMemgraphTool, BaseTool):
    """Tool for retrieving constraint information from Memgraph."""

    name: str = ShowConstraintInfoTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = ShowConstraintInfoTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Optional[Type[BaseModel]] = None
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Run the tool to get constraint information."""

        constraint_info = ShowConstraintInfoTool(
            db=self.db,
        )
        result = constraint_info.call({})

        return result


class RunShowIndexInfoTool(BaseMemgraphTool, BaseTool):
    """Tool for retrieving index information from Memgraph."""

    name: str = ShowIndexInfoTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = ShowIndexInfoTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Optional[Type[BaseModel]] = None
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Run the tool to get index information."""

        index_info = ShowIndexInfoTool(
            db=self.db,
        )
        result = index_info.call({})

        return result


class RunShowConfigTool(BaseMemgraphTool, BaseTool):
    """Tool for retrieving configuration information from Memgraph."""

    name: str = ShowConfigTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = ShowConfigTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Optional[Type[BaseModel]] = None
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Run the tool to get configuration information."""

        config_info = ShowConfigTool(
            db=self.db,
        )
        result = config_info.call({})

        return result


class RunShowTriggersTool(BaseMemgraphTool, BaseTool):
    """Tool for retrieving trigger information from Memgraph."""

    name: str = ShowTriggersTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = ShowTriggersTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Optional[Type[BaseModel]] = None
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        """Run the tool to get trigger information."""

        trigger_info = ShowTriggersTool(
            db=self.db,
        )
        result = trigger_info.call({})

        return result


class _BetweennessCentralityToolInput(BaseModel):
    """
    Input schema for the Betweenness Centrality Memgraph tool.
    """

    isDirectionIgnored: bool = Field(
        True,
        description="Set to false to consider the direction of relationships. Default is true.",
    )
    limit: int = Field(
        10, description="Limit the number of nodes to return. Default is 10."
    )


class RunBetweennessCentralityTool(BaseMemgraphTool, BaseTool):
    """Tool for running Betweenness Centrality on Memgraph."""

    name: str = BetweennessCentralityTool(db=None).get_name()
    """The name that is passed to the model when performing tool calling."""

    description: str = BetweennessCentralityTool(db=None).get_description()
    """The description that is passed to the model when performing tool calling."""

    args_schema: Type[BaseModel] = _BetweennessCentralityToolInput
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        isDirectionIgnored: bool = True,
        limit: int = 10,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        return BetweennessCentralityTool(
            db=self.db,
        ).call({"isDirectionIgnored": isDirectionIgnored, "limit": limit})
