"""Memgraph tools."""

from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field

from langchain_memgraph.graphs.memgraph import Memgraph


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


class QueryMemgraphTool(BaseMemgraphTool, BaseTool):  # type: ignore[override]
    """Tool for querying Memgraph.

    Setup:
        Install ``langchain-memgraph`` and make sure Memgraph is running.

        .. code-block:: bash
            pip install -U langchain-memgraph

    Instantiation:
        .. code-block:: python

            tool = QueryMemgraphTool(
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

    # TODO: Add any other init params for the tool.
    # param1: Optional[str]
    # """param1 determines foobar"""

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> List[Dict[str, Any]]:
        return self.db.query(query)
