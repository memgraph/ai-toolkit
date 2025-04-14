from typing import Any, Dict, List
from core.api.tool import BaseTool
from core.api.memgraph import MemgraphClient


class ShowConstraintInfo(BaseTool):
    """
    Tool for showing constraint information from Memgraph.
    """
    def __init__(self, db: MemgraphClient):
        super().__init__(
            name="show_constraint_info",
            description="Shows constraint information from a Memgraph database",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the SHOW CONSTRAINT INFO query and return the results."""
        constraint_info = self.db.query("SHOW CONSTRAINT INFO")
        return constraint_info

    def close(self):
        """Close the database connection."""
        self.db.close()
