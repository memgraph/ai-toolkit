from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class ShowIndexInfoTool(BaseTool):
    """
    Tool for showing index information in Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="show_index_info",
            description="Shows index information from a Memgraph database",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the SHOW INDEX INFO query and return the results."""
        index_info = self.db.query("SHOW INDEX INFO")
        return index_info
