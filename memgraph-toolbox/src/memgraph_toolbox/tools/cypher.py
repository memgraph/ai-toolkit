from typing import Any

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class CypherTool(BaseTool):
    """
    Tool for running arbitrary Cypher queries on Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="run_cypher_query",
            description="Executes a Cypher query on a Memgraph database",
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

    def call(self, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        """Execute the provided Cypher query and return the results."""
        query = arguments["query"]
        try:
            return self.db.query(query)
        except Exception as e:
            return [{"error": f"Failed to execute query: {e!s}"}]
