"""Memgraph tools."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from core.api.memgraph import MemgraphClient
from core.tools.schema import ShowSchemaInfoTool

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


class QueryMemgraphTool(BaseMemgraphTool, BaseTool):  # type: ignore[override]
    """Tool for querying Memgraph.

    Setup:
        Install ``langchain-memgraph`` and make sure Memgraph is running.

        .. code-block:: bash
            pip install -U langchain-memgraph

    Instantiation:
        .. code-block:: python

            tool = QueryMemgraphTool(
                memgraph_client=memgraph_client
            )

    Invocation with args:
        .. code-block:: python

            tool.invoke({"query" : "MATCH (n) RETURN n LIMIT 1"})

        .. code-block:: python

            # Output of invocation
            # List[Dict[str, Any]
            [
                {
                    "n": {
                        "name": "Alice",
                        "age": 30
                    }
                }
            ]

    """  # noqa: E501

    name: str = "memgraph_cypher_query"
    """The name that is passed to the model when performing tool calling."""

    description: str = (
        "Tool is used to query Memgraph via Cypher query and returns the result."
    )
    """The description that is passed to the model when performing tool calling."""

    args_schema: Type[BaseModel] = _QueryMemgraphToolInput
    """The schema that is passed to the model when performing tool calling."""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        return self.db.query(query)


class ShowSchemaInfoTool(BaseMemgraphTool, BaseTool):  # type: ignore[override]
    """Tool for retrieving schema information from Memgraph.

    Setup:
        Install ``langchain-memgraph`` and make sure Memgraph is running.

        .. code-block:: bash
            pip install -U langchain-memgraph

    Instantiation:
        .. code-block:: python

            tool = ShowSchemaInfoTool(
                db=memgraph_client
            )

    Invocation:
        .. code-block:: python

            result = tool.invoke({})

        .. code-block:: python

            # Output of invocation
            # List[Dict[str, Any]]
            [
                {
                    "node_labels": ["Person", "Movie"],
                    "edge_types": ["ACTED_IN", "DIRECTED"],
                    "node_properties": ["name", "age", "title"],
                    "edge_properties": ["role", "year"]
                }
            ]
    """  # noqa: E501

    name: str = "memgraph_show_schema"
    """The name that is passed to the model when performing tool calling."""

    description: str = (
        "Tool is used to retrieve schema information from Memgraph database."
    )
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