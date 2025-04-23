from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class ShowConfigTool(BaseTool):
    """
    Tool for showing configuration information from Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="show_config",
            description="Shows configuration information from a Memgraph database",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the SHOW CONFIG query and return the results."""
        config_info = self.db.query("SHOW CONFIG")
        return config_info
