from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class CypherTool(BaseTool):
    """
    Tool for running arbitrary Cypher queries on Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="run_cypher",
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

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the provided Cypher query and return the results."""
        query = arguments["query"]
        result = self.db.query(query)
        return [dict(record) for record in result]
