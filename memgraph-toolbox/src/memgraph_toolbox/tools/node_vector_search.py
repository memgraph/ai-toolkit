from typing import Any, Dict, List, Optional

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class NodeVectorSearchTool(BaseTool):
    """
    Tool for performing vector similarity search on nodes in Memgraph.
    """

    def __init__(self, db: Memgraph):
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
                },
                "required": ["index_name", "query_vector"],
            },
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute vector similarity search and return the results."""
        index_name = arguments["index_name"]
        query_vector = arguments["query_vector"]
        limit = arguments.get("limit", 10)

        query = f"""
            CALL vector_search.search(\"{index_name}\", {limit}, {query_vector}) YIELD * RETURN *;
        """
        try:
            results = self.db.query(query, {})
            records = []
            for record in results:
                node = record["node"]
                properties = {k: v for k, v in node.items() if k != "embedding"}
                node_data = {
                    # TODO(gitbuda): Not possible to return all node info
                    # because of serialization done under api.memgraph:85
                    # because the record.data() is returning a dict of only
                    # properties
                    # (https://neo4j.com/docs/api/python-driver/current/api.html#neo4j.Record.data)
                    # FIX under https://github.com/memgraph/ai-toolkit/pull/68.
                    "properties": properties,
                    "distance": record["distance"],
                }
                records.append(node_data)
            return records
        except Exception as e:
            return [
                {
                    "error": "Unexpected failure during the NodeVectorSearch tool call.",
                    "traceback": str(e),
                }
            ]
