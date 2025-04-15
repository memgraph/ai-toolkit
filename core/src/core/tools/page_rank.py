from typing import Any, Dict, List
from core.api.tool import BaseTool
from core.api.memgraph import MemgraphClient


class PageRankTool(BaseTool):
    """
    Tool for calculating PageRank on a graph in Memgraph.
    """
    def __init__(self, db: MemgraphClient):
        super().__init__(
            name="pagerank",
            description="Calculates PageRank on a graph in Memgraph",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of nodes to return ",
                        "default": 10
                    },
                },
                "required": []
            }
        )
        self.db = db

    #TODO:(@antejavor) This will fail if user is not running Memgraph Mage since memgraph does not have this built in. 
    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the PageRank algorithm and return the results."""
        limit = arguments.get("limit", 20)

        query = (
            f"""
            CALL pagerank.get()
            YIELD node, rank
            RETURN node, rank
            ORDER BY rank DESC LIMIT {limit}
            """
        )
        pagerank_results = self.db.query(query)

        return pagerank_results

    def close(self):
        """Close the database connection."""
        self.db.close()
