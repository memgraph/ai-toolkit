# This is SHOW SCHEMA INFO tool from Memgraph 
from typing import Any, Dict, List
from core.api.tool import BaseTool
from core.api.memgraph import MemgraphClient


class ShowSchemaInfo(BaseTool):
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

    def call(self, arguments: Dict[str, Any]) -> List[Any]:
        schema_info = self.db.query("SHOW SCHEMA INFO")
        
        # Convert the schema information to a list of dictionaries
        schema_info_list = [dict(record) for record in schema_info]
        
        # Return the schema information
        return schema_info_list

    def close(self):
        self.db.close()

