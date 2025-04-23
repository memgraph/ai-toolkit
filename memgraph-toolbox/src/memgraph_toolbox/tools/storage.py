from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class ShowStorageInfoTool(BaseTool):
    """
    Tool for showing storage information from Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="show_storage_info",
            description="Shows storage information from a Memgraph database",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the SHOW STORAGE INFO query and return the results."""
        storage_info = self.db.query("SHOW STORAGE INFO")
        return storage_info
