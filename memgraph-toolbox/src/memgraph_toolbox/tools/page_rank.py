from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class PageRankTool(BaseTool):
    """
    Tool for calculating PageRank on a graph in Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="page_rank",
            description="Calculates PageRank on a graph in Memgraph",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Number of nodes to return ",
                        "default": 10,
                    },
                },
                "required": [],
            },
        )
        self.db = db

    # TODO:(@antejavor) This will fail if user is not running Memgraph Mage since memgraph does not have this built in.
    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the PageRank algorithm and return the results."""
        limit = arguments.get("limit", 20)

        query = f"""
            CALL pagerank.get()
            YIELD node, rank
            RETURN node, rank
            ORDER BY rank DESC LIMIT {limit}
            """
        pagerank_results = self.db.query(query)

        return pagerank_results
