# This is SHOW SCHEMA INFO tool from Memgraph 
from typing import Any, Dict, List
from core.api.tool import BaseTool
from core.api.memgraph import MemgraphClient


class ShowSchemaInfoTool(BaseTool):
    """
    Tool for showing schema information from Memgraph.
    """
    def __init__(self, db: MemgraphClient):
        super().__init__(
            name="show_schema_info",
            description="Shows schema information from a Memgraph database",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        self.db = db

    def call(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        schema_info = self.db.query("SHOW SCHEMA INFO")
        return schema_info

    def close(self):
        self.db.close()

