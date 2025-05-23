from mcp.server.fastmcp import FastMCP

from memgraph_toolbox.api.memgraph import Memgraph
from memgraph_toolbox.tools.cypher import CypherTool
from memgraph_toolbox.tools.config import ShowConfigTool
from memgraph_toolbox.tools.index import ShowIndexInfoTool
from memgraph_toolbox.tools.constraint import ShowConstraintInfoTool
from memgraph_toolbox.tools.schema import ShowSchemaInfoTool
from memgraph_toolbox.tools.storage import ShowStorageInfoTool
from memgraph_toolbox.tools.trigger import ShowTriggersTool
from memgraph_toolbox.tools.betweenness_centrality import BetweennessCentralityTool
from memgraph_toolbox.tools.page_rank import PageRankTool
from memgraph_toolbox.utils.logging import logger_init


from typing import Any, Dict, List

# Configure logging
logger = logger_init("mcp-memgraph")

# Initialize FastMCP server
mcp = FastMCP("mcp-memgraph")

MEMGRAPH_URL = "bolt://localhost:7687"
MEMGRAPH_USER = ""
MEMGRAPH_PASSWORD = ""

# Initialize Memgraph client
db = Memgraph(
    url=MEMGRAPH_URL,
    username=MEMGRAPH_USER,
    password=MEMGRAPH_PASSWORD,
)


@mcp.tool()
def run_query(query: str) -> List[Dict[str, Any]]:
    """Run a Cypher query on Memgraph"""
    logger.info(f"Running query: {query}")
    try:
        result = CypherTool(db=db).call({"query": query})
        return result
    except Exception as e:
        return [f"Error running query: {str(e)}"]


@mcp.tool()
def get_configuration() -> List[Dict[str, Any]]:
    """Get Memgraph configuration information"""
    logger.info("Fetching Memgraph configuration...")
    try:
        config = ShowConfigTool(db=db).call({})
        return config
    except Exception as e:
        return [f"Error fetching configuration: {str(e)}"]


@mcp.tool()
def get_index() -> List[Dict[str, Any]]:
    """Get Memgraph index information"""
    logger.info("Fetching Memgraph index...")
    try:
        index = ShowIndexInfoTool(db=db).call({})
        return index
    except Exception as e:
        return [f"Error fetching index: {str(e)}"]


@mcp.tool()
def get_constraint() -> List[Dict[str, Any]]:
    """Get Memgraph constraint information"""
    logger.info("Fetching Memgraph constraint...")
    try:
        constraint = ShowConstraintInfoTool(db=db).call({})
        return constraint
    except Exception as e:
        return [f"Error fetching constraint: {str(e)}"]


@mcp.tool()
def get_schema() -> List[Dict[str, Any]]:
    """Get Memgraph schema information"""
    logger.info("Fetching Memgraph schema...")
    try:
        schema = ShowSchemaInfoTool(db=db).call({})
        return schema
    except Exception as e:
        return [f"Error fetching schema: {str(e)}"]


@mcp.tool()
def get_storage() -> List[Dict[str, Any]]:
    """Get Memgraph storage information"""
    logger.info("Fetching Memgraph storage...")
    try:
        storage = ShowStorageInfoTool(db=db).call({})
        return storage
    except Exception as e:
        return [f"Error fetching storage: {str(e)}"]


@mcp.tool()
def get_triggers() -> List[Dict[str, Any]]:
    """Get Memgraph triggers information"""
    logger.info("Fetching Memgraph triggers...")
    try:
        triggers = ShowTriggersTool(db=db).call({})
        return triggers
    except Exception as e:
        return [f"Error fetching triggers: {str(e)}"]


@mcp.tool()
def get_betweenness_centrality() -> List[Dict[str, Any]]:
    """Get betweenness centrality information"""
    logger.info("Fetching betweenness centrality...")
    try:
        betweenness = BetweennessCentralityTool(db=db).call({})
        return betweenness
    except Exception as e:
        return [f"Error fetching betweenness centrality: {str(e)}"]


@mcp.tool()
def get_page_rank() -> List[Dict[str, Any]]:
    """Get page rank information"""
    logger.info("Fetching page rank...")
    try:
        page_rank = PageRankTool(db=db).call({})
        return page_rank
    except Exception as e:
        return [f"Error fetching page rank: {str(e)}"]
