from typing import Any, Dict, List

from ..api.memgraph import Memgraph
from ..api.tool import BaseTool


class ShowSchemaInfoTool(BaseTool):
    """
    Tool for showing schema information from Memgraph.
    """

    def __init__(self, db: Memgraph):
        super().__init__(
            name="show_schema_info",
            description="Shows schema information from a Memgraph database",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        schema_info = self.db.query("SHOW SCHEMA INFO")
        return schema_info
