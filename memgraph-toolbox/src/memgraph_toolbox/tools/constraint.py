from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class ShowConstraintInfoTool(BaseTool):
    """
    Tool for showing constraint information from Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="show_constraint_info",
            description="Shows constraint information from a Memgraph database",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the SHOW CONSTRAINT INFO query and return the results."""
        constraint_info = self.db.query("SHOW CONSTRAINT INFO")
        return constraint_info
