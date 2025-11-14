from fastmcp import FastMCP

from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.tools.config import ShowConfigTool
from memgraph_toolbox.tools.index import ShowIndexInfoTool
from memgraph_toolbox.tools.constraint import ShowConstraintInfoTool
from memgraph_toolbox.tools.schema import ShowSchemaInfoTool
from memgraph_toolbox.tools.storage import ShowStorageInfoTool
from memgraph_toolbox.tools.trigger import ShowTriggersTool
from memgraph_toolbox.tools.betweenness_centrality import (
    BetweennessCentralityTool,
)
from memgraph_toolbox.tools.page_rank import PageRankTool
from memgraph_toolbox.tools.node_neighborhood import NodeNeighborhoodTool
from memgraph_toolbox.tools.node_vector_search import NodeVectorSearchTool
from memgraph_toolbox.utils.logger import logger_init

from typing import Any, Dict, List
import re

from mcp_memgraph.config import get_memgraph_config, get_mcp_config

# Get configuration instances
memgraph_config = get_memgraph_config()
mcp_config = get_mcp_config()

# Configure logging
logger = logger_init("mcp-memgraph")

# Initialize FastMCP server
mcp = FastMCP("mcp-memgraph")

# Read-only mode flag (from config)
READ_ONLY_MODE = mcp_config.read_only

# Patterns for write operations in Cypher
WRITE_PATTERNS = [
    r"\bCREATE\b",
    r"\bMERGE\b",
    r"\bDELETE\b",
    r"\bREMOVE\b",
    r"\bSET\b",
    r"\bDROP\b",
    r"\bCREATE\s+INDEX\b",
    r"\bDROP\s+INDEX\b",
    r"\bCREATE\s+CONSTRAINT\b",
    r"\bDROP\s+CONSTRAINT\b",
]


def is_write_query(query: str) -> bool:
    """Check if a Cypher query contains write operations"""
    query_upper = query.upper()
    for pattern in WRITE_PATTERNS:
        if re.search(pattern, query_upper):
            return True
    return False


# Initialize Memgraph client using configuration
logger.info(
    "Connecting to Memgraph db '%s' at %s with user '%s'",
    memgraph_config.database,
    memgraph_config.url,
    memgraph_config.username,
)
logger.info("Read-only mode: %s", READ_ONLY_MODE)

db = Memgraph(**memgraph_config.get_client_config())


@mcp.tool()
def run_query(query: str) -> List[Dict[str, Any]]:
    """Run a Cypher query on Memgraph. Write operations are blocked if
    server is in read-only mode."""
    logger.info("Running query: %s", query)

    # Check if query is a write operation in read-only mode
    if READ_ONLY_MODE and is_write_query(query):
        logger.warning("Write operation blocked in read-only mode: %s", query)
        return [
            {
                "error": "Write operations are not allowed in read-only mode",
                "query": query,
                "mode": "read-only",
                "hint": "Set MCP_READ_ONLY=false to enable write operations",
            }
        ]

    try:
        result = CypherTool(db=db).call({"query": query})
        return result
    except Exception as e:
        return [{"error": f"Error running query: {str(e)}"}]


@mcp.tool()
def get_configuration() -> List[Dict[str, Any]]:
    """Get Memgraph configuration information"""
    logger.info("Fetching Memgraph configuration...")
    try:
        config = ShowConfigTool(db=db).call({})
        return config
    except Exception as e:
        return [{"error": f"Error fetching configuration: {str(e)}"}]


@mcp.tool()
def get_index() -> List[Dict[str, Any]]:
    """Get Memgraph index information"""
    logger.info("Fetching Memgraph index...")
    try:
        index = ShowIndexInfoTool(db=db).call({})
        return index
    except Exception as e:
        return [{"error": f"Error fetching index: {str(e)}"}]


@mcp.tool()
def get_constraint() -> List[Dict[str, Any]]:
    """Get Memgraph constraint information"""
    logger.info("Fetching Memgraph constraint...")
    try:
        constraint = ShowConstraintInfoTool(db=db).call({})
        return constraint
    except Exception as e:
        return [{"error": f"Error fetching constraint: {str(e)}"}]


@mcp.tool()
def get_schema() -> List[Dict[str, Any]]:
    """Get Memgraph schema information"""
    logger.info("Fetching Memgraph schema...")
    try:
        schema = ShowSchemaInfoTool(db=db).call({})
        return schema
    except Exception as e:
        return [{"error": f"Error fetching schema: {str(e)}"}]


@mcp.tool()
def get_storage() -> List[Dict[str, Any]]:
    """Get Memgraph storage information"""
    logger.info("Fetching Memgraph storage...")
    try:
        storage = ShowStorageInfoTool(db=db).call({})
        return storage
    except Exception as e:
        return [{"error": f"Error fetching storage: {str(e)}"}]


@mcp.tool()
def get_triggers() -> List[Dict[str, Any]]:
    """Get Memgraph triggers information"""
    logger.info("Fetching Memgraph triggers...")
    try:
        triggers = ShowTriggersTool(db=db).call({})
        return triggers
    except Exception as e:
        return [{"error": f"Error fetching triggers: {str(e)}"}]


@mcp.tool()
def get_betweenness_centrality() -> List[Dict[str, Any]]:
    """Get betweenness centrality information"""
    logger.info("Fetching betweenness centrality...")
    try:
        betweenness = BetweennessCentralityTool(db=db).call({})
        return betweenness
    except Exception as e:
        return [{"error": f"Error fetching betweenness centrality: {str(e)}"}]


@mcp.tool()
def get_page_rank() -> List[Dict[str, Any]]:
    """Get page rank information"""
    logger.info("Fetching page rank...")
    try:
        page_rank = PageRankTool(db=db).call({})
        return page_rank
    except Exception as e:
        return [{"error": f"Error fetching page rank: {str(e)}"}]


@mcp.tool()
def get_node_neighborhood(
    node_id: str, max_distance: int = 1, limit: int = 100
) -> List[Dict[str, Any]]:
    """Find nodes within a specified distance from a given node"""
    logger.info(
        "Finding neighborhood for node %s with max distance %s",
        node_id,
        max_distance,
    )
    try:
        neighborhood = NodeNeighborhoodTool(db=db).call(
            {"node_id": node_id, "max_distance": max_distance, "limit": limit}
        )
        return neighborhood
    except Exception as e:
        return [{"error": f"Error finding node neighborhood: {str(e)}"}]


@mcp.tool()
def search_node_vectors(
    index_name: str, query_vector: List[float], limit: int = 10
) -> List[Dict[str, Any]]:
    """Perform vector similarity search on nodes in Memgraph"""
    logger.info("Performing vector search on index %s with limit %s", index_name, limit)
    try:
        vector_search = NodeVectorSearchTool(db=db).call(
            {
                "index_name": index_name,
                "query_vector": query_vector,
                "limit": limit,
            }
        )
        return vector_search
    except Exception as e:
        return [{"error": f"Error performing vector search: {str(e)}"}]
