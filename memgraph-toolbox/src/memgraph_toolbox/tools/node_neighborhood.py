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
                        "description": "Maximum distance (hops) to search from the starting node. Default is 2.",
                        "default": 2,
                    },
                    "relationship_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of relationship types to include in the search. If empty, all types are included.",
                        "default": [],
                    },
                    "node_labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of node labels to include in the search. If empty, all labels are included.",
                        "default": [],
                    },
                    "include_paths": {
                        "type": "boolean",
                        "description": "Whether to include the paths from start node to each neighbor. Default is false.",
                        "default": False,
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
        max_distance = arguments.get("max_distance", 2)
        relationship_types = arguments.get("relationship_types", [])
        node_labels = arguments.get("node_labels", [])
        include_paths = arguments.get("include_paths", False)
        limit = arguments.get("limit", 100)

        # Build relationship type filter
        rel_filter = ""
        if relationship_types:
            rel_types_str = ", ".join([f"'{rt}'" for rt in relationship_types])
            rel_filter = f"WHERE type(r) IN [{rel_types_str}]"

        # Build node label filter
        label_filter = ""
        if node_labels:
            label_conditions = [
                f"ANY(label IN labels(n) WHERE label IN {node_labels})"
                for _ in node_labels
            ]
            label_filter = f"AND {' AND '.join(label_conditions)}"

        if include_paths:
            # Query with paths included
            query = f"""
            MATCH path = (start)-[*1..{max_distance}]-{rel_filter}(neighbor)
            WHERE start.element_id = $node_id {label_filter}
            WITH neighbor, path, length(path) as distance
            RETURN DISTINCT neighbor, distance, path
            ORDER BY distance, neighbor.element_id
            LIMIT $limit
            """
        else:
            # Query without paths (more efficient)
            query = f"""
            MATCH (start)-[*1..{max_distance}]-{rel_filter}(neighbor)
            WHERE start.element_id = $node_id {label_filter}
            WITH DISTINCT neighbor, length(shortestPath((start)-[*]-(neighbor))) as distance
            RETURN neighbor, distance
            ORDER BY distance, neighbor.element_id
            LIMIT $limit
            """

        params = {"node_id": node_id, "limit": limit}

        try:
            results = self.db.query(query, params)

            # Process results to extract relevant information
            processed_results = []
            for record in results:
                node_data = {
                    "node_id": (
                        record["neighbor"].element_id
                        if hasattr(record["neighbor"], "element_id")
                        else str(record["neighbor"])
                    ),
                    "labels": (
                        list(record["neighbor"].labels)
                        if hasattr(record["neighbor"], "labels")
                        else []
                    ),
                    "properties": (
                        dict(record["neighbor"])
                        if hasattr(record["neighbor"], "__iter__")
                        else {}
                    ),
                    "distance": record["distance"],
                }

                if include_paths and "path" in record:
                    node_data["path"] = str(record["path"])

                processed_results.append(node_data)

            return processed_results

        except Exception as e:
            return [{"error": f"Failed to find neighborhood: {str(e)}"}]
