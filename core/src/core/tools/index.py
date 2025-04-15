from typing import Any, Dict, List
from core.api.tool import BaseTool
from core.api.memgraph import MemgraphClient


class ShowIndexInfoTool(BaseTool):
    """
    Tool for showing index information in Memgraph.
    """
    def __init__(self, db: MemgraphClient):
        super().__init__(
            name="show_index_info",
            description="Shows index information from a Memgraph database",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the SHOW INDEX INFO query and return the results."""
        storage_info = self.db.query("SHOW STORAGE INFO")  
        return storage_info

    def close(self):
        """Close the database connection."""
        self.db.close()
