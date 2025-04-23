from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class BetweennessCentralityTool(BaseTool):
    """
    Tool for calculating betweenness centrality on a graph in Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="run_betweenness_centrality",
            description=(
                "Calculates betweenness centrality for nodes in a graph. "
                "Betweenness centrality measures the importance of a node based on how frequently "
                "it lies on the shortest paths between other nodes, indicating its influence on "
                "information flow within the network."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "isDirectionIgnored": {
                        "type": "boolean",
                        "description": "Set to false to consider the direction of relationships. Default is true.",
                        "default": True,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Limit the number of nodes to return. Default is 10.",
                        "default": 10,
                    },
                },
                "required": [],
            },
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the betweenness centrality algorithm and return the results."""
        is_direction_ignored = arguments.get("isDirectionIgnored", True)
        limit = arguments.get("limit", 10)

        query = (
            "CALL betweenness_centrality.get($directed, True) "
            "YIELD node, betweenness_centrality "
            "RETURN node, betweenness_centrality "
            "ORDER BY betweenness_centrality DESC "
            "LIMIT $limit"
        )
        params = {"directed": not is_direction_ignored, "limit": limit}

        results = self.db.query(query, params)

        # Convert the results to a list of dictionaries
        return results
