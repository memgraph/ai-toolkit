from typing import Any, Dict, List
from core.api.tool import BaseTool
from core.api.memgraph import MemgraphClient


class ShowConfig(BaseTool):
    """
    Tool for showing configuration information from Memgraph.
    """
    def __init__(self, db: MemgraphClient):
        super().__init__(
            name="show_config",
            description="Shows configuration information from a Memgraph database",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the SHOW CONFIG query and return the results."""
        config_info = self.db.query("SHOW CONFIG")
        return config_info

    def close(self):
        """Close the database connection."""
        self.db.close()
