# This is SHOW SCHEMA INFO tool from Memgraph 
from neo4j import GraphDatabase
from typing import Any, Dict, List
from core.api.tool import BaseTool

class ShowSchemaInfo(BaseTool):
    def __init__(self, uri: str, db: GraphDatabase.driver):
        super().__init__(
            name="show_schema_info",
            description="Shows schema information from a Memgraph database",
            input_schema={
                "type": "object",
                "properties": {},
                "required": []
            }
        )
        self.driver: GraphDatabase.driver = db

    async def call(self, arguments: Dict[str, Any]) -> List[Any]:
        # Get the schema information from the database
        schema_info = self.driver.session().run("SHOW SCHEMA INFO")
        
        # Convert the schema information to a list of dictionaries
        schema_info_list = [dict(record) for record in schema_info]
        
        # Return the schema information
        return schema_info_list

    def close(self):
        self.driver.close()

