from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class NodeNeighborhoodTool(BaseTool):
    """
    Tool for finding nodes within a specified neighborhood distance in Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="node_neighborhood",
            description=(
                "Finds nodes within a specified distance from a given node. "
                "This tool explores the graph neighborhood around a starting node, "
                "returning all nodes and relationships found within the specified radius."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The ID of the starting node to find neighborhood around",
                    },
                    "max_distance": {
                        "type": "integer",
                        "description": "Maximum distance (hops) to search from the starting node. Default is 1.",
                        "default": 1,
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of nodes to return. Default is 100.",
                        "default": 100,
                    },
                },
                "required": ["node_id"],
            },
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the neighborhood search and return the results."""
        node_id = arguments["node_id"]
        max_distance = arguments.get("max_distance", 1)
        limit = arguments.get("limit", 100)

        query = f"""MATCH (n)-[r*..{max_distance}]-(m) WHERE id(n) = {node_id} RETURN DISTINCT m LIMIT {limit};"""
        try:
            results = self.db.query(query, {})
            processed_results = []
            for record in results:
                node_data = record["m"];
                properties = {k: v for k, v in node_data.items()}
                processed_results.append(properties)
            return processed_results
        except Exception as e:
            return [{"error": f"Failed to find neighborhood: {str(e)}"}]
