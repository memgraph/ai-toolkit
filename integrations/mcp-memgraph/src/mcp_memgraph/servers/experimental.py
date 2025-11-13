from fastmcp import FastMCP
from memgraph_toolbox.utils.logger import logger_init

from typing import Any, Dict, List

from mcp_memgraph.config import get_memgraph_config, get_mcp_config

# Get configuration instances (reusing existing config)
memgraph_config = get_memgraph_config()
mcp_config = get_mcp_config()

# Configure logging
logger = logger_init("mcp-memgraph-experimental")

# Initialize FastMCP server for experimental features
mcp = FastMCP("mcp-memgraph-experimental")

logger.info("🧪 Experimental MCP server initialized")
logger.info("Read-only mode: %s", mcp_config.read_only)


@mcp.tool()
def experimental_query() -> List[Dict[str, Any]]:
    """Experimental query tool that returns hardcoded data for testing
    new features"""
    logger.info("Running experimental query...")

    # Return some hardcoded graph-like data
    return [
        {
            "node_id": 1,
            "name": "Experimental Node A",
            "type": "TestNode",
            "properties": {
                "status": "experimental",
                "version": "0.1.0",
                "description": "This is a hardcoded test node",
            },
        },
        {
            "node_id": 2,
            "name": "Experimental Node B",
            "type": "TestNode",
            "properties": {
                "status": "experimental",
                "version": "0.1.0",
                "description": "Another hardcoded test node",
            },
        },
        {
            "relationship": {
                "from": 1,
                "to": 2,
                "type": "CONNECTS_TO",
                "properties": {"weight": 0.95, "experimental": True},
            }
        },
    ]
