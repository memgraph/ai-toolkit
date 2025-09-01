from typing import Any, Dict, List, Optional

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class NodeVectorSearchTool(BaseTool):
    """
    Tool for performing vector similarity search on nodes in Memgraph.
    """

    def __init__(self, db: Memgraph):
        # TODO(gitbuda): The problem is that we have index_name instead of label :thinking:
        super().__init__(
            name="node_vector_search",
            description="Performs vector similarity search on nodes in Memgraph using cosine similarity",
            input_schema={
                "type": "object",
                "properties": {
                    "index_name": {
                        "type": "string",
                        "description": "Name of the index to use for the vector search",
                    },
                    "node_label": {
                        "type": "string",
                        "description": "Label of the nodes to search (e.g., 'Person', 'Document')",
                    },
                    "vector_property": {
                        "type": "string",
                        "description": "Property name containing the vector embeddings",
                        "default": "embedding",
                    },
                    "query_vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Query vector to search for similarity",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of similar nodes to return",
                        "default": 10,
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity threshold (0.0 to 1.0)",
                        "default": 0.5,
                    },
                    "return_properties": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of node properties to return in results",
                        "default": ["id", "name"],
                    },
                },
                "required": ["index_name", "query_vector"],
            },
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute vector similarity search and return the results."""
        index_name = arguments["index_name"]
        node_label = arguments["node_label"]
        vector_property = arguments.get("vector_property", "embedding")
        query_vector = arguments["query_vector"]
        limit = arguments.get("limit", 10)
        similarity_threshold = arguments.get("similarity_threshold", 0.5)
        return_properties = arguments.get("return_properties", ["id", "name"])

        # Create the Cypher query for vector similarity search
        query = f"""
            CALL vector_search.search({index_name}, {limit}, {query_vector}) YIELD * RETURN *;"
        """
        # TODO(gitbuda): Pass other params.
        params = {
            "query_vector": query_vector,
            "similarity_threshold": similarity_threshold,
            "limit": limit,
        }

        try:
            results = self.db.query(query, params)
            records = []
            for record in results:
                node = record["node"]
                properties = {k: v for k, v in node.items() if k != "embedding"}
                node_data = {
                    "distance": record["distance"],
                    "id": node.element_id,
                    "labels": list(node.labels),
                    "properties": properties,
                }
                records.append(node_data)
            return records
        except Exception as e:
            # TODO: Fallback to a simpler query if something fails...
            return [
                {"error": "Unexpected failure during the NodeVectorSearch tool call."}
            ]
