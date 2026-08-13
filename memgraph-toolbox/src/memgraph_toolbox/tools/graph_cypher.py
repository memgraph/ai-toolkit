from typing import Any

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool
from ..utils.serialization import project_graph


class GraphCypherTool(BaseTool):
    """
    Tool for running a Cypher query and returning a graph projection.

    Unlike :class:`CypherTool`, which returns the raw tabular rows with graph
    entities flattened to bare property maps, this tool returns a normalized
    ``{"nodes": [...], "relationships": [...]}`` structure with node and edge
    identity, labels, relationship types, and endpoints preserved.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="run_cypher_graph",
            description="Executes a Cypher query and returns a graph projection of nodes and relationships",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Cypher query to execute",
                    }
                },
                "required": ["query"],
            },
        )
        self.db = db

    def call(self, arguments: dict[str, Any]) -> dict[str, Any] | list[dict[str, Any]]:
        """Execute the query and project the result into nodes and relationships."""
        query = arguments["query"]
        try:
            records = self.db.query_raw(query)
            return project_graph(records)
        except Exception as e:
            return [{"error": f"Failed to execute query: {e!s}"}]
