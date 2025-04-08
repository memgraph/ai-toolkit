from core.api.memgraph import MemgraphClient
from typing import Dict, List, Any
from core.api.tool import BaseTool
from core.tools.schema import ShowSchemaInfo
import asyncio
from core.api.toolkit import Toolkit

async def main():
    # Initialize the Neo4j driver
    uri = "bolt://localhost:7687"  # Default Memgraph URI
    user = "memgraph"  # Default Memgraph user
    password = "memgraph"  # Default Memgraph password

    memgraph_client = MemgraphClient(uri, user, password)
    
    toolkit = Toolkit()
    
    # Create and add the schema info tool
    schema_tool = ShowSchemaInfo(memgraph_client=memgraph_client)
    toolkit.add(schema_tool)
    
    print("Added tools:", toolkit.list_tools())
    
    # Get and call the schema tool
    tool = toolkit.get_tool("show_schema_info")
    result = await tool.call({})
    print(f"Schema information from '{tool.name}':", result)
    
    # Clean up
    schema_tool.close()

if __name__ == "__main__":
    asyncio.run(main())
